from stdatamodels.jwst import datamodels

from jwst.pathloss import pathloss
from jwst.stpipe import Step

__all__ = ["PathLossStep"]


class PathLossStep(Step):
    """
    PathLossStep: Apply the path loss correction to a science exposure.

    Pathloss depends on the centering of the source in the aperture if the
    source is a point source.
    """

    class_alias = "pathloss"

    spec = """
        inverse = boolean(default=False)    # Invert the operation
        source_type = string(default=None)  # Process as specified source type
        user_slit_loc = float(default=None)   # User-provided correction to MIRI LRS source location
    """  # noqa: E501

    reference_file_types = ["pathloss"]

    def process(self, model_input):
        """
        Execute the pathloss step.

        Parameters
        ----------
        model_input : DataModel
            The input datamodel to be corrected.

        Returns
        -------
        result : DataModel
            The result of the pathloss calibration step.
        """
        # Open the input data model
        with datamodels.open(model_input) as input_model:
            if self.use_correction_pars:
                correction_pars = self.correction_pars
                pathloss_model = None
            else:
                correction_pars = None

                # Get the name of the pathloss reference file to use
                self.pathloss_name = self.get_reference_file(input_model, "pathloss")
                self.log.info(f"Using PATHLOSS reference file {self.pathloss_name}")

                # Check for a valid reference file
                if self.pathloss_name == "N/A":
                    self.log.warning("No PATHLOSS reference file found")
                    self.log.warning("Pathloss step will be skipped")
                    result = input_model.copy()
                    result.meta.cal_step.pathloss = "SKIPPED"
                    return result

                # Open the pathloss ref file data model
                if input_model.meta.exposure.type.upper() in ["MIR_LRS-FIXEDSLIT"]:
                    pathloss_model = datamodels.MirLrsPathlossModel(self.pathloss_name)
                else:
                    pathloss_model = datamodels.PathlossModel(self.pathloss_name)

            # Do the pathloss correction
            result, self.correction_pars = pathloss.do_correction(
                input_model,
                pathloss_model,
                inverse=self.inverse,
                source_type=self.source_type,
                correction_pars=correction_pars,
                user_slit_loc=self.user_slit_loc,
            )

            if pathloss_model:
                pathloss_model.close()

        return result
