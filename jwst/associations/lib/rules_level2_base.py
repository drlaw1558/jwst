"""Base classes which define the Level2 Associations."""

import copy
import logging
import re
from collections import deque
from os.path import split
from pathlib import Path

from jwst.associations import Association, libpath
from jwst.associations.exceptions import AssociationNotValidError
from jwst.associations.lib.acid import ACID
from jwst.associations.lib.constraint import (
    Constraint,
    SimpleConstraint,
)
from jwst.associations.lib.dms_base import (
    IMAGE2_NONSCIENCE_EXP_TYPES,
    IMAGE2_SCIENCE_EXP_TYPES,
    PRODUCT_NAME_DEFAULT,
    SPEC2_SCIENCE_EXP_TYPES,
    Constraint_TargetAcq,
    DMSAttrConstraint,
    DMSBaseMixin,
)
from jwst.associations.lib.member import Member
from jwst.associations.lib.process_list import ListCategory
from jwst.associations.lib.prune import prune
from jwst.associations.lib.rules_level3_base import _EMPTY, DMS_Level3_Base
from jwst.associations.lib.rules_level3_base import Utility as Utility_Level3
from jwst.associations.lib.utilities import getattr_from_list, getattr_from_list_nofail
from jwst.associations.registry import RegistryMarker
from jwst.lib.suffix import remove_suffix

# Configure logging
logger = logging.getLogger(__name__)

__all__ = [
    "_EMPTY",
    "ASN_SCHEMA",
    "AsnMixin_Lv2Image",
    "AsnMixin_Lv2Nod",
    "AsnMixin_Lv2Imprint",
    "AsnMixin_Lv2Special",
    "AsnMixin_Lv2Spectral",
    "AsnMixin_Lv2WFSS",
    "Constraint_Background",
    "Constraint_Base",
    "Constraint_ExtCal",
    "Constraint_Image_Nonscience",
    "Constraint_Image_Science",
    "Constraint_Imprint",
    "Constraint_Imprint_Special",
    "Constraint_Mode",
    "Constraint_Single_Science",
    "Constraint_Special",
    "Constraint_Spectral_Science",
    "Constraint_Target",
    "DMSLevel2bBase",
    "DMSAttrConstraint",
    "Utility",
]

# The schema that these associations must adhere to.
ASN_SCHEMA = RegistryMarker.schema(libpath() / "asn_schema_jw_level2b.json")

# Flag to exposure type
FLAG_TO_EXPTYPE = {
    "background": "background",
}

# File templates
_DMS_POOLNAME_REGEX = r"jw(\d{5})_(\d{3})_(\d{8}[Tt]\d{6})_pool"
_LEVEL1B_REGEX = r"(?P<path>.+)(?P<type>_uncal)(?P<extension>\..+)"

# Key that uniquely identifies items.
KEY = "expname"


class DMSLevel2bBase(DMSBaseMixin, Association):
    """Basic class for DMS Level2 associations."""

    # Set the validation schema
    schema_file = ASN_SCHEMA.schema

    # Attribute values that are indicate the
    # attribute is not specified.
    INVALID_VALUES = _EMPTY

    def __init__(self, *args, **kwargs):
        super(DMSLevel2bBase, self).__init__(*args, **kwargs)

        # Initialize validity checks
        self.validity.update(
            {
                "has_science": {
                    "validated": False,
                    "check": lambda member: member["exptype"] == "science",
                },
                "allowed_candidates": {"validated": False, "check": self.validate_candidates},
            }
        )
        # Other presumptions on the association
        if "constraints" not in self.data:
            self.data["constraints"] = "No constraints"
        if "asn_type" not in self.data:
            self.data["asn_type"] = "user_built"
        if "asn_id" not in self.data:
            self.data["asn_id"] = "a3001"
        if "asn_pool" not in self.data:
            self.data["asn_pool"] = "none"

    def get_exposure_type(self, item, default="science"):
        """
        General Level 2 override of exposure type definition.

        The exposure type definition is overridden from the default
        for the following cases:

        - 'psf' -> 'science'

        Parameters
        ----------
        item : dict
            The pool entry to determine the exposure type of
        default : str or None
            The default exposure type.
            If None, routine will raise LookupError

        Returns
        -------
        exposure_type
            Always what is defined as `default`
        """
        self.original_exposure_type = super(DMSLevel2bBase, self).get_exposure_type(
            item, default=default
        )
        if self.original_exposure_type == "psf":
            return default
        return self.original_exposure_type

    def members_by_type(self, member_type):
        """
        Get list of members by their exposure type.

        Returns
        -------
        list
            List of members.
        """
        member_type = member_type.lower()
        try:
            members = self.current_product["members"]
        except KeyError:
            result = []
        else:
            result = [member for member in members if member_type == member["exptype"].lower()]

        return result

    def has_science(self):
        """
        Check association for a science member.

        Returns
        -------
        bool
            True if it does.
        """
        limit_reached = len(self.members_by_type("science")) >= 1
        return limit_reached

    def dms_product_name(self):
        """
        Define product name.

        Returns
        -------
        str
            The product name.
        """
        try:
            science = self.members_by_type("science")[0]
        except IndexError:
            return PRODUCT_NAME_DEFAULT

        try:
            path_expname = Path(science["expname"])
            science_path = path_expname.parent / path_expname.stem
        except Exception:
            return PRODUCT_NAME_DEFAULT

        no_suffix_path, separator = remove_suffix(str(science_path))
        return no_suffix_path

    def make_member(self, item):
        """
        Create a member from the item.

        Parameters
        ----------
        item : dict
            The item to create member from.

        Returns
        -------
        member : Member
            The member
        """
        # Set exposure error status.
        try:
            exposerr = item["exposerr"]
        except KeyError:
            exposerr = None

        # Create the member.
        # The various `is_item_xxx` methods are used to determine whether the name
        # should represent the form of the data product containing all integrations.
        member = Member(
            {
                "expname": Utility.rename_to_level2a(
                    item["filename"],
                    use_integrations=(
                        self.is_item_coron(item)
                        |
                        # NIS_AMI used to use rate files;
                        # updated to use rateints
                        self.is_item_ami(item)
                        | self.is_item_tso(item)
                    ),
                ),
                "exptype": self.get_exposure_type(item),
                "exposerr": exposerr,
            },
            item=item,
        )

        return member

    def _init_hook(self, item):
        """Post-check and pre-add initialization."""
        self.data["target"] = item["targetid"]
        self.data["program"] = "{:0>5s}".format(item["program"])
        self.data["asn_pool"] = Path(item.meta["pool_file"]).name
        self.data["constraints"] = str(self.constraints)
        self.data["asn_id"] = self.acid.id
        self.new_product(self.dms_product_name())

    def _add(self, item):
        """
        Add item to this association.

        Parameters
        ----------
        item : dict
            The item to be adding.
        """
        member = self.make_member(item)
        if self.is_member(member):
            return
        members = self.current_product["members"]
        members.append(member)
        self.update_validity(member)

        # Update association state due to new member
        self.update_asn()

    def _add_items(self, items, meta=None, product_name_func=None, acid="o999", **kwargs):
        """
        Force adding items to the association.

        Parameters
        ----------
        items : [object[, ...]]
            A list of items to make members of the association.
        meta : dict
            A dict to be merged into the association meta information.
            The following are suggested to be assigned:
                - `asn_type`
                    The type of association.
                - `asn_rule`
                    The rule which created this association.
                - `asn_pool`
                    The pool from which the exposures came from
                - `program`
                    Originating observing program
        product_name_func : func
            Used if product name is 'undefined' using
            the class's procedures.
        acid : str
            The association candidate id to use. Since Level2
            associations require it, one must be specified.

        Notes
        -----
        This is a low-level shortcut into adding members, such as file names,
        to an association. All defined shortcuts and other initializations are
        by-passed, resulting in a potentially unusable association.

        `product_name_func` is used to define the product names instead of
        the default methods. The call signature is:

            product_name_func(item, idx)

        where `item` is each item being added and `idx` is the count of items.
        """
        if meta is None:
            meta = {}

        # Setup association candidate.
        if acid.startswith("o"):
            ac_type = "observation"
        elif acid.startswith("c"):
            ac_type = "background"
        else:
            raise ValueError(
                f'Invalid association id specified: "{acid}"\n\tMust be of form "oXXX" or "c1XXX"'
            )
        self._acid = ACID((acid, ac_type))

        # set the default exptype
        exptype = "science"

        for idx, item in enumerate(items, start=1):
            self.new_product()
            members = self.current_product["members"]
            if isinstance(item, tuple):
                expname = item[0]
            else:
                expname = item

            # check to see if kwargs are passed and if exptype is given
            if kwargs:
                if "with_exptype" in kwargs:
                    if item[1]:
                        exptype = item[1]
                    else:
                        exptype = "science"
            member = Member({"expname": expname, "exptype": exptype}, item=item)
            members.append(member)
            self.update_validity(member)
            self.update_asn()

            # If a product name function is given, attempt
            # to use.
            if product_name_func is not None:
                try:
                    self.current_product["name"] = product_name_func(item, idx)
                except Exception:
                    logger.debug(
                        "Attempted use of product_name_func failed. Default product name used."
                    )

        self.data.update(meta)
        self.current_sequence = next(self.sequence)

    def update_asn(self):
        """Update association info based on current members."""
        super(DMSLevel2bBase, self).update_asn()
        self.current_product["name"] = self.dms_product_name()

    def validate_candidates(self, _member):
        """
        Allow only OBSERVATION or BACKGROUND candidates.

        Parameters
        ----------
        _member : member
            Member being added; ignored.

        Returns
        -------
        bool
            True if candidate is OBSERVATION, BACKGROUND and at least one
            member is background; otherwise false.
        """
        # If an observation, then we're good.
        if self.acid.type.lower() == "observation":
            return True

        # If a background, check that there is a background
        # exposure
        if self.acid.type.lower() == "background":
            for entry in self.current_product["members"]:
                if entry["exptype"].lower() == "background":
                    return True

        # If not background member, or some other candidate type,
        # fail.
        return False

    def __repr__(self):
        try:
            file_name, json_repr = self.ioregistry["json"].dump(self)
        except Exception:
            return str(self.__class__)
        return json_repr

    def __str__(self):
        """
        Create human readable version of the association.

        Returns
        -------
        str
            Human-readable string version of association.
        """
        result = [f"Association {self.asn_name:s}"]

        # Parameters of the association
        result.append(
            "    Parameters:"
            "        Product type: {asn_type:s}"
            "        Rule:         {asn_rule:s}"
            "        Program:      {program:s}"
            "        Target:       {target:s}"
            "        Pool:         {asn_pool:s}".format(
                asn_type=getattr(self.data, "asn_type", "indetermined"),
                asn_rule=getattr(self.data, "asn_rule", "indetermined"),
                program=getattr(self.data, "program", "indetermined"),
                target=getattr(self.data, "target", "indetermined"),
                asn_pool=getattr(self.data, "asn_pool", "indetermined"),
            )
        )

        result.append(f"        {str(self.constraints):s}")

        # Products of the association
        for product in self.data["products"]:
            result.append("\t{} with {} members".format(product["name"], len(product["members"])))

        # That's all folks
        result.append("\n")
        return "\n".join(result)


@RegistryMarker.utility
class Utility:
    """Utility functions that understand DMS Level 2 associations."""

    @staticmethod
    @RegistryMarker.callback("finalize")
    def finalize(associations):
        """
        Check validity and duplications in an association list.

        Parameters
        ----------
        associations : [association[, ...]]
            List of associations.

        Returns
        -------
        finalized_associations : [association[, ...]]
            The validated list of associations.
        """
        finalized_asns = []
        lv2_asns = []
        for asn in associations:
            if isinstance(asn, DMSLevel2bBase):
                finalized = asn.finalize()
                if finalized is not None:
                    lv2_asns.extend(finalized)
            else:
                finalized_asns.append(asn)

        # Having duplicate Level 2 associations is expected.
        lv2_asns = prune(lv2_asns)

        # Ensure sequencing is correct.
        Utility_Level3.resequence(lv2_asns)

        return finalized_asns + lv2_asns

    @staticmethod
    def merge_asns(associations):
        """
        Merge level2 associations.

        Parameters
        ----------
        associations : [asn(, ...)]
            Associations to search for merging.

        Returns
        -------
        associations : [association(, ...)]
            List of associations, some of which may be merged.
        """
        others = []
        lv2_asns = []
        for asn in associations:
            if isinstance(asn, DMSLevel2bBase):
                lv2_asns.append(asn)
            else:
                others.append(asn)

        lv2_asns = Utility._merge_asns(lv2_asns)

        return others + lv2_asns

    @staticmethod
    def rename_to_level2a(level1b_name, use_integrations=False):
        """
        Rename a Level 1b Exposure to another level.

        Parameters
        ----------
        level1b_name : str
            The Level 1b exposure name.
        use_integrations : bool
            Use 'rateints' instead of 'rate' as the suffix.

        Returns
        -------
        str
            The Level 2a name.
        """
        match = re.match(_LEVEL1B_REGEX, level1b_name)
        if match is None or match.group("type") != "_uncal":
            logger.warning(
                f"Item FILENAME='{level1b_name}' is not a Level 1b name. "
                "Cannot transform to Level 2a."
            )
            return level1b_name

        suffix = "rate"
        if use_integrations:
            suffix = "rateints"
        level2a_name = "".join([match.group("path"), "_", suffix, match.group("extension")])
        return level2a_name

    @staticmethod
    def resequence(*args, **kwargs):
        """
        Resequence the numbers to conform to level 3 associations.

        Returns
        -------
        list[association, ...] or None
            If associations provided, resequenced order of
            provided associations.
        """
        return Utility_Level3.resequence(*args, **kwargs)

    @staticmethod
    def sort_by_candidate(asns):
        """
        Sort associations by candidate.

        Parameters
        ----------
        asns : [Association[,...]]
            List of associations

        Returns
        -------
        sorted_by_candidate : [Associations[,...]]
            New list of the associations sorted.

        Notes
        -----
        The current definition of candidates allows strictly alphabetical
        sorting:
        aXXXX > cXXXX > oXXX

        If this changes, a comparison function will need be implemented
        """
        return sorted(asns, key=lambda asn: asn["asn_id"])

    @staticmethod
    def _merge_asns(asns):
        """
        Merge associations by `asn_type` and `asn_id`.

        Parameters
        ----------
        asns : [asn(, ...)]
            Associations to search for merging.

        Returns
        -------
        associations : [association(, ...)]
            List of associations, some of which may be merged.
        """
        merged = {}
        for asn in asns:
            idx = "_".join([asn["asn_type"], asn["asn_id"]])
            try:
                current_asn = merged[idx]
            except KeyError:
                merged[idx] = asn
                current_asn = asn
            for product in asn["products"]:
                merge_occurred = False
                for current_product in current_asn["products"]:
                    if product["name"] == current_product["name"]:
                        member_names = {member["expname"] for member in product["members"]}
                        current_member_names = [
                            member["expname"] for member in current_product["members"]
                        ]
                        new_names = member_names.difference(current_member_names)
                        new_members = [
                            member
                            for member in product["members"]
                            if member["expname"] in new_names
                        ]
                        current_product["members"].extend(new_members)
                        merge_occurred = True
                if not merge_occurred:
                    current_asn["products"].append(product)

        merged_asns = list(merged.values())
        return merged_asns


# Basic constraints
class Constraint_Base(Constraint):
    """Select on program and instrument."""

    def __init__(self):
        super(Constraint_Base, self).__init__(
            [
                DMSAttrConstraint(name="program", sources=["program"]),
                DMSAttrConstraint(
                    name="is_tso",
                    sources=["tsovisit"],
                    required=False,
                    force_unique=True,
                ),
            ]
        )


class Constraint_Background(DMSAttrConstraint):
    """Select backgrounds."""

    def __init__(self):
        super(Constraint_Background, self).__init__(
            sources=["bkgdtarg"],
            force_unique=False,
            name="background",
            force_reprocess=ListCategory.EXISTING,
            only_on_match=True,
        )


class Constraint_ExtCal(Constraint):
    """
    Remove any nis_extcals from the associations.

    They are NOT to receive level-2b or level-3 processing.
    """

    def __init__(self):
        super(Constraint_ExtCal, self).__init__(
            [DMSAttrConstraint(name="exp_type", sources=["exp_type"], value="nis_extcal")],
            reduce=Constraint.notany,
        )


class Constraint_Imprint(Constraint):
    """Select on imprint exposures."""

    def __init__(self):
        super(Constraint_Imprint, self).__init__(
            [DMSAttrConstraint(name="imprint", sources=["is_imprt"])],
            reprocess_on_match=True,
            work_over=ListCategory.EXISTING,
        )


class Constraint_Imprint_Special(Constraint):
    """Select on imprint exposures with mosaic tile number."""

    def __init__(self, association=None):
        # If an association is not provided, the check for original
        # exposure type is ignored.
        if association is None:

            def sources(_item):
                return "not imprint"
        else:

            def sources(_item):
                return association.original_exposure_type

        super(Constraint_Imprint_Special, self).__init__(
            [
                DMSAttrConstraint(
                    name="imprint",
                    sources=["is_imprt"],
                    force_reprocess=ListCategory.EXISTING,
                    only_on_match=True,
                ),
                DMSAttrConstraint(
                    name="mosaic_tile",
                    sources=["mostilno"],
                ),
                SimpleConstraint(
                    value="imprint",
                    sources=sources,
                    test=lambda v1, v2: v1 != v2,
                    force_unique=False,
                ),
            ],
        )


class Constraint_Mode(Constraint):
    """Select on instrument and optical path."""

    def __init__(self):
        super(Constraint_Mode, self).__init__(
            [
                DMSAttrConstraint(name="instrument", sources=["instrume"]),
                DMSAttrConstraint(name="detector", sources=["detector"]),
                DMSAttrConstraint(name="opt_elem", sources=["filter", "band"]),
                DMSAttrConstraint(
                    name="opt_elem2",
                    sources=["pupil", "grating"],
                    required=False,
                ),
                DMSAttrConstraint(
                    name="fxd_slit",
                    sources=["fxd_slit"],
                    required=False,
                ),
                DMSAttrConstraint(
                    name="subarray",
                    sources=["subarray"],
                    required=False,
                ),
                DMSAttrConstraint(
                    name="channel",
                    sources=["channel"],
                    required=False,
                ),
                Constraint(
                    [
                        DMSAttrConstraint(sources=["detector"], value="nirspec"),
                        DMSAttrConstraint(sources=["filter"], value="opaque"),
                    ],
                    reduce=Constraint.notany,
                ),
                Constraint(
                    [
                        DMSAttrConstraint(sources=["detector"], value="nrcblong"),
                        DMSAttrConstraint(sources=["exp_type"], value="nrc_tsgrism"),
                    ],
                    reduce=Constraint.notall,
                ),
                Constraint(
                    [
                        DMSAttrConstraint(
                            sources=["visitype"],
                            value=".+wfsc.+",
                        ),
                    ],
                    reduce=Constraint.notany,
                ),
                DMSAttrConstraint(
                    name="slit",
                    sources=["fxd_slit"],
                    required=False,
                ),
            ]
        )


class Constraint_Image_Science(DMSAttrConstraint):
    """Select on science images."""

    def __init__(self):
        super(Constraint_Image_Science, self).__init__(
            name="exp_type", sources=["exp_type"], value="|".join(IMAGE2_SCIENCE_EXP_TYPES)
        )


class Constraint_Image_Nonscience(Constraint):
    """Select on non-science images."""

    def __init__(self):
        super(Constraint_Image_Nonscience, self).__init__(
            [
                Constraint_TargetAcq(),
                DMSAttrConstraint(
                    name="non_science",
                    sources=["exp_type"],
                    value="|".join(IMAGE2_NONSCIENCE_EXP_TYPES),
                ),
                Constraint(
                    [
                        DMSAttrConstraint(
                            name="exp_type", sources=["exp_type"], value="nrs_msaspec"
                        ),
                        DMSAttrConstraint(sources=["msastate"], value="primarypark_allopen"),
                        DMSAttrConstraint(sources=["grating"], value="mirror"),
                    ]
                ),
            ],
            reduce=Constraint.any,
        )


class Constraint_Single_Science(Constraint):
    """
    Allow only single science exposure.

    Notes
    -----
    The `has_science_fn` is further wrapped in a lambda function
    to provide a closure. Otherwise if the function is a bound method,
    that method may end up pointing to an instance that is not calling
    this constraint.
    """

    def __init__(self, has_science_fn, exposure_type_fn, **sc_kwargs):
        """
        Initialize a new single science constraint.

        Parameters
        ----------
        has_science_fn : func
            Function to determine whether the association has a science member already.
            No arguments are provided

        exposure_type_fn : func
            Function to determine the association exposure type of the item.
            Should take a single argument of item.

        **sc_kwargs : dict
            Keyword arguments to pass to the parent class `Constraint`
        """
        super(Constraint_Single_Science, self).__init__(
            [
                SimpleConstraint(
                    value=True,
                    sources=lambda item: exposure_type_fn(item) == "science",
                ),
                SimpleConstraint(
                    value=False,
                    sources=lambda _item: has_science_fn(),
                ),
            ],
            **sc_kwargs,
        )


class Constraint_Special(DMSAttrConstraint):
    """Select on backgrounds and other auxiliary images."""

    def __init__(self):
        super(Constraint_Special, self).__init__(
            name="is_special",
            sources=["bkgdtarg", "is_psf"],
        )


class Constraint_Spectral_Science(Constraint):
    """
    Select on spectral science.

    Parameters
    ----------
    exclude_exp_types : [exp_type[, ...]]
        List of exposure types to not consider from
        from the general list.
    """

    def __init__(self, exclude_exp_types=None):
        if exclude_exp_types is None:
            general_science = SPEC2_SCIENCE_EXP_TYPES
        else:
            general_science = set(SPEC2_SCIENCE_EXP_TYPES).symmetric_difference(exclude_exp_types)

        super(Constraint_Spectral_Science, self).__init__(
            [
                DMSAttrConstraint(
                    name="exp_type", sources=["exp_type"], value="|".join(general_science)
                )
            ],
            reduce=Constraint.any,
        )


class Constraint_Target(Constraint):
    """Select on target id."""

    def __init__(self):
        constraints = [
            Constraint(
                [
                    DMSAttrConstraint(
                        name="acdirect",
                        sources=["asn_candidate"],
                        value=r"\[\('c\d{4}', 'direct_image'\)\]",
                    ),
                    SimpleConstraint(name="target", sources=lambda _item: "000"),
                ]
            ),
            DMSAttrConstraint(
                name="target",
                sources=["targetid"],
            ),
        ]
        super(Constraint_Target, self).__init__(constraints, reduce=Constraint.any)


# ---------------------------------------------
# Mixins to define the broad category of rules.
# ---------------------------------------------
class AsnMixin_Lv2Image:
    """Level 2 Image association base."""

    def _init_hook(self, item):
        """Post-check and pre-add initialization."""
        super(AsnMixin_Lv2Image, self)._init_hook(item)
        self.data["asn_type"] = "image2"


class AsnMixin_Lv2Imprint:
    """Level 2 association handling for matching imprint images."""

    def prune_imprints(self):
        """
        Prune extra imprint exposures from the association members.

        First, check for imprints that match the background flag
        for the "science" member.  Any included imprints that do
        not match the background flag are left in the association
        without further checks.

        Among these imprints, check to see if any have target IDs that
        match the science member. If not, use all imprints for further
        matches.  If so, use only the matching ones for further checks;
        remove the non-matching imprints.

        If there are more imprints than science members remaining, and
        if any of the remaining imprints match the science dither position
        index, discard any imprints that do not match the dither position.
        """
        # Only one product for Lv2 associations
        members = self["products"][0]["members"]

        # Check for science and imprint members
        science = []
        imprint_sci = []
        imprint_bkg = []
        science_is_background = False
        science_targets = set()
        for member in members:
            try:
                target = member.item["targetid"]
                bkgdtarg = member.item["bkgdtarg"]
            except KeyError:
                # ignore any members with missing data -
                # no pruning will happen in this case
                continue

            if member["exptype"] == "science":
                science.append(member)
                science_targets.add(str(target))
                if bkgdtarg == "t":
                    science_is_background = True
            elif member["exptype"] == "imprint":
                # Store imprints by background status
                if bkgdtarg == "t":
                    imprint_bkg.append(member)
                else:
                    imprint_sci.append(member)

        # Only check the imprints that match the "science" members
        if science_is_background:
            imprints_to_check = imprint_bkg
        else:
            imprints_to_check = imprint_sci

        # If there are multiple targets in the imprints to check,
        # and if one of them matches the science data, keep only that one
        imprints_matching_target = []
        imprints_not_matching_target = []
        for imprint_exp in imprints_to_check:
            try:
                target = imprint_exp.item["targetid"]
            except KeyError:
                # Imprint does not match target if it is missing a target ID
                imprints_not_matching_target.append(imprint_exp)
                continue
            if str(target) in science_targets:
                imprints_matching_target.append(imprint_exp)
            else:
                imprints_not_matching_target.append(imprint_exp)
        if len(imprints_matching_target) > 0:
            imprints_to_check = imprints_matching_target
            for imprint_exp in imprints_not_matching_target:
                members.remove(imprint_exp)

        # If 1 or more science and more imprints than science,
        # discard any imprints that don't match the science dither index
        if 1 <= len(science) < len(imprints_to_check):
            imprint_to_keep = set()
            for science_exp in science:
                if "dithptin" not in science_exp.item:
                    continue
                for imprint_exp in imprints_to_check:
                    if "dithptin" not in imprint_exp.item:
                        continue
                    if imprint_exp.item["dithptin"] == science_exp.item["dithptin"]:
                        imprint_to_keep.add(imprint_exp["expname"])

            # if there were any matching imprints, remove the extras
            if len(imprint_to_keep) > 0:
                for imprint_exp in imprints_to_check:
                    if imprint_exp["expname"] not in imprint_to_keep:
                        members.remove(imprint_exp)

    def finalize(self):
        """
        Finalize the association.

        For some spectrographic modes, imprint images are taken alongside
        the science data, the background data, or both.  If there are
        extra imprints in the association, we should keep only the best
        matches to the science data.

        Returns
        -------
        associations: [association[, ...]] or None
            List of fully-qualified associations that this association
            represents.
            `None` if a complete association cannot be produced.
        """
        if self.is_valid:
            self.prune_imprints()
        return super().finalize()


class AsnMixin_Lv2Nod:
    """
    Associations that need to create nodding associations.

    For some spectrographic modes, background spectra are taken by
    nodding between different slits, or internal slit positions.
    The main associations rules will collect all the exposures of all the nods
    into a single association. Then, on finalization, this one association
    is split out into many associations, where each nod is, in turn, treated
    as science, and the other nods are treated as background.
    """

    @staticmethod
    def nod_background_overlap(science_item, background_item):
        """
        Check that a candidate background nod will not overlap with science.

        For NIRSpec fixed slit or MOS data, this returns True if the
        background candidate shares a primary dither point with the
        science or if the target ID does not match between science
        and background candidate.

        In addition, for NIRSpec fixed slit exposures taken with slit
        S1600A1 in a 5-point nod pattern, this returns True when the
        background candidate is in the next closest primary position
        to the science or if the target ID does not match between
        science and background candidate.

        For any other data, this function returns False (no overlap).

        Parameters
        ----------
        science_item : member
            The science member.
        background_item : member
            The background member.

        Returns
        -------
        bool
            True if overlap is present, false otherwise.
        """
        # Get exp_type, needed for any data:
        # if not present, return False
        try:
            exptype = str(science_item["exp_type"]).lower()
        except KeyError:
            return False

        # Return False for any non-FS or MOS data
        if exptype not in ["nrs_fixedslit", "nrs_msaspec"]:
            return False

        # Check for target ID - if present,
        # it must match for either FS or MOS
        try:
            sci_target_id = str(science_item["targetid"]).lower()
            bkg_target_id = str(background_item["targetid"]).lower()
        except KeyError:
            sci_target_id = None
            bkg_target_id = None
        if sci_target_id != bkg_target_id:
            # Report overlap - different activities contain
            # different sources that may overlap, even if not
            # at the same primary nod position.
            return True

        # Get pattern number values, needed for FS or MOS
        try:
            numdthpt = int(science_item["numdthpt"])
            patt_num = int(science_item["patt_num"])
            bkg_num = int(background_item["patt_num"])
        except (KeyError, ValueError):
            numdthpt = 0
            patt_num = None
            bkg_num = None

        # Get subpxpts (or old alternate 'subpxpns')
        try:
            subpx = int(science_item["subpxpts"])
        except (KeyError, ValueError):
            try:
                subpx = int(science_item["subpxpns"])
            except (KeyError, ValueError):
                subpx = None
        try:
            bkg_subpx = int(background_item["subpxpts"])
        except (KeyError, ValueError):
            try:
                bkg_subpx = int(background_item["subpxpns"])
            except (KeyError, ValueError):
                bkg_subpx = None

        # If values not found, return False for MOS, True for FS
        if None in [patt_num, bkg_num, subpx, bkg_subpx] or subpx == 0 or bkg_subpx == 0:
            if exptype == "nrs_fixedslit":
                # These values are required for background
                # candidates for FS: report an overlap.
                return True
            else:
                # For MOS, it's okay to go ahead and use them.
                # Report no overlap.
                return False

        # Check for primary point overlap
        sci_prim_dithpt = (patt_num - 1) // subpx
        bkg_prim_dithpt = (bkg_num - 1) // bkg_subpx
        if sci_prim_dithpt == bkg_prim_dithpt:
            # Primary points are the same - overlap is present
            return True

        # Primary points are not the same -
        # no further check needed for MOS
        if exptype == "nrs_msaspec":
            return False

        # Get slit name, needed only for FS S1600A1 check
        try:
            slit = str(science_item["fxd_slit"]).lower()
        except KeyError:
            slit = None

        # Background is currently only expected to overlap severely for
        # 5 point dithers with fixed slit S1600A1.  Only the nearest
        # dithers to the science observation need to be excluded.
        if (
            exptype == "nrs_fixedslit"
            and slit == "s1600a1"
            and numdthpt // subpx == 5
            and abs(sci_prim_dithpt - bkg_prim_dithpt) <= 1
        ):
            return True

        # For anything else, return False
        return False

    def make_nod_asns(self):
        """
        Make background nod Associations.

        For observing modes, such as NIRSpec MSA, exposures can be
        nodded, such that the object is in a different position in the
        slitlet. The association creation simply groups these all
        together as a single association, all exposures marked as
        `science`. When complete, this method will create separate
        associations each exposure becoming the single science
        exposure, and the other exposures then become `background`.

        Returns
        -------
        associations : [association[, ...]]
            List of new associations to be used in place of
            the current one.
        """
        for product in self["products"]:
            members = product["members"]

            # Split out the science vs. non-science
            # The non-science exposures will get attached
            # to every resulting association.
            science_exps = [member for member in members if member["exptype"] == "science"]
            nonscience_exps = [member for member in members if member["exptype"] != "science"]

            # Create new associations for each science, using
            # the other science as background.
            results = []
            for science_exp in science_exps:
                asn = copy.deepcopy(self)
                asn.data["products"] = None

                product_name = remove_suffix(Path(split(science_exp["expname"])[1]).stem)[0]
                asn.new_product(product_name)
                new_members = asn.current_product["members"]
                new_members.append(science_exp)

                for other_science in science_exps:
                    if other_science["expname"] != science_exp["expname"]:
                        # check for likely overlap between science
                        # and background candidate
                        overlap = self.nod_background_overlap(science_exp.item, other_science.item)
                        if not overlap:
                            now_background = Member(other_science)
                            now_background["exptype"] = "background"
                            new_members.append(now_background)

                new_members += nonscience_exps

                if asn.is_valid:
                    results.append(asn)

            return results

    def finalize(self):
        """
        Finalize association.

        For some spectrographic modes, background spectra are taken by
        nodding between different slits, or internal slit positions.
        The main associations rules will collect all the exposures of all the nods
        into a single association. Then, on finalization, this one association
        is split out into many associations, where each nod is, in turn, treated
        as science, and the other nods are treated as background.

        Returns
        -------
        associations : [association[, ...]] or None
            List of fully-qualified associations that this association
            represents.
            `None` if a complete association cannot be produced.
        """
        if self.is_valid:
            return self.make_nod_asns()
        else:
            return None


class AsnMixin_Lv2Special:
    """
    Process special and non-science exposures as science.

    Attributes
    ----------
    original_exposure_type : str
        The original exposure type of what is referred to as the "science" member
    """

    original_exposure_type = None

    def get_exposure_type(self, item, default="science"):
        """
        Override to force exposure type to always be science.

        The only case where this should not happen is if the association
        already has its science, and the current item is an imprint. Leave
        that item as an imprint.

        Parameters
        ----------
        item : dict
            The pool entry to determine the exposure type of
        default : str or None
            The default exposure type.
            If None, routine will raise LookupError

        Returns
        -------
        exposure_type
            Always what is defined as `default`
        """
        self.original_exposure_type = super(AsnMixin_Lv2Special, self).get_exposure_type(
            item, default=default
        )
        if self.has_science():
            if self.original_exposure_type == "imprint":
                return "imprint"
        return default


class AsnMixin_Lv2Spectral(DMSLevel2bBase):
    """Level 2 Spectral association base."""

    def _init_hook(self, item):
        """Post-check and pre-add initialization."""
        super(AsnMixin_Lv2Spectral, self)._init_hook(item)
        self.data["asn_type"] = "spec2"


class AsnMixin_Lv2WFSS:
    """Utility and overrides for WFSS associations."""

    def add_catalog_members(self):
        """Add catalog and direct image member based on direct image members."""
        directs = self.members_by_type("direct_image")
        if not directs:
            raise AssociationNotValidError(
                f"{self.__class__.__name__} has no required direct image exposures"
            )

        sciences = self.members_by_type("science")
        if not sciences:
            raise AssociationNotValidError(
                f"{self.__class__.__name__} has no required science exposure"
            )
        science = sciences[0]

        # Remove all direct images from the association.
        members = self.current_product["members"]
        direct_idxs = [
            idx for idx, member in enumerate(members) if member["exptype"] == "direct_image"
        ]
        deque(list.pop(members, idx) for idx in sorted(direct_idxs, reverse=True))

        # Add the Level3 catalog, direct image, and segmentation map members
        self.direct_image = self.find_closest_direct(science, directs)
        lv3_direct_image_root = DMS_Level3_Base._dms_product_name(self)  # noqa: SLF001
        members.append(
            Member({"expname": lv3_direct_image_root + "_i2d.fits", "exptype": "direct_image"})
        )
        members.append(
            Member({"expname": lv3_direct_image_root + "_cat.ecsv", "exptype": "sourcecat"})
        )
        members.append(
            Member({"expname": lv3_direct_image_root + "_segm.fits", "exptype": "segmap"})
        )

    def finalize(self):
        """
        Finalize the association.

        For WFSS, this involves taking all the direct image exposures,
        determine which one is first after last science exposure,
        and creating the catalog name from that image.

        Returns
        -------
        associations : [association[, ...]] or None
            List of fully-qualified associations that this association
            represents.
            `None` if a complete association cannot be produced.
        """
        try:
            self.add_catalog_members()
        except AssociationNotValidError as err:
            logger.debug("%s: %s", self.__class__.__name__, str(err))
            return None

        return super(AsnMixin_Lv2WFSS, self).finalize()

    def get_exposure_type(self, item, default="science"):
        """
        Modify exposure type depending on dither pointing index.

        If an imaging exposure has been found, treat it as a direct image.

        Parameters
        ----------
        item : member
            The item to pull exposure type from.
        default : str
            The default value if no exposure type is present, defaults to "science".

        Returns
        -------
        str
            The exposure type of the item.
        """
        exp_type = super(AsnMixin_Lv2WFSS, self).get_exposure_type(item, default)
        if exp_type == "science" and item["exp_type"] in ["nis_image", "nrc_image"]:
            exp_type = "direct_image"

        return exp_type

    @staticmethod
    def find_closest_direct(science, directs):
        """
        Find the direct image that is closest to the science.

        Closeness is defined as number difference in the exposure sequence number,
        as defined in the column EXPSPCIN.

        Parameters
        ----------
        science : dict
            The science member to compare against

        directs : [dict[,...]]
            The available direct members

        Returns
        -------
        closest : dict
            The direct image that is the "closest"
        """
        closest = directs[0]  # If the search fails, just use the first.
        try:
            expspcin = int(getattr_from_list(science.item, ["expspcin"], _EMPTY)[1])
        except KeyError:
            # If exposure sequence cannot be determined, just fall through.
            logger.debug("Science exposure %s has no EXPSPCIN defined.", science)
        else:
            min_diff = 9999  # Initialize to an invalid value.
            for direct in directs:
                try:
                    direct_expspcin = int(getattr_from_list(direct.item, ["expspcin"], _EMPTY)[1])
                except KeyError:
                    # Try the next one.
                    logger.debug("Direct image %s has no EXPSPCIN defined.", direct)
                    continue
                diff = direct_expspcin - expspcin
                if diff < min_diff and diff > 0:
                    min_diff = diff
                    closest = direct
        return closest

    def get_opt_element(self):
        """
        Get string representation of the optical elements.

        Returns
        -------
        opt_elem: str
            The Level3 Product name representation
            of the optical elements.

        Notes
        -----
        This is an override for the method in `DMSBaseMixin`.
        The optical element is retrieved from the chosen direct image
        found in `self.direct_image`, determined in the `self.finalize`
        method.
        """
        item = self.direct_image.item
        opt_elems = []
        for keys in [["filter", "band"], ["pupil", "grating"]]:
            opt_elem = getattr_from_list_nofail(item, keys, _EMPTY)[1]
            if opt_elem:
                opt_elems.append(opt_elem)
        opt_elems.sort(key=str.lower)
        full_opt_elem = "-".join(opt_elems)
        if full_opt_elem == "":
            full_opt_elem = "clear"

        return full_opt_elem
