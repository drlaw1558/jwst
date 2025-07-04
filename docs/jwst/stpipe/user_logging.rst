.. _user-logging:

=======
Logging
=======

Log messages are emitted from each Step at different levels of
importance.  The levels used are the standard ones for Python (from
least important to most important:

#. DEBUG
#. INFO
#. WARNING
#. ERROR
#. CRITICAL

By default, only messages of type WARNING or higher are displayed.
This can be controlled by providing a logging configuration file.

Logging configuration
=====================

A logging configuration file can be provided to customize what is
logged.

A logging configuration file is searched for in the following places.
The first one found is used *in its entirety* and all others are
ignored:

#. The file specified with the ``--logcfg`` option to the
   ``strun`` script.

#. The file specified with the ``logcfg`` keyword to a
   .call() execution of a Step or Pipeline.

#. A file called ``stpipe-log.cfg`` in the current working
   directory.

#. ``~/stpipe-log.cfg``

#. ``/etc/stpipe-log.cfg``

The logging configuration file is in the standard ini-file format.
The file should contain a single section called ``[*]`` that contains:

#. ``level``: The level at and above which logging messages will be
   displayed.  May be one of (from least important to most
   important): DEBUG, INFO, WARNING, ERROR or CRITICAL.

#. ``break_level``: The level at and above which logging messages
   will cause an exception to be raised.  For instance, if you
   would rather stop execution at the first ERROR message (rather
   than continue), set ``break_level`` to ``ERROR``.

#. ``handler``: Defines where log messages are to be sent.  By
   default, they are sent to stderr.  However, one may also
   specify:

     - ``file:filename.log`` to send the log messages to the given
       file.

     - ``append:filename.log`` to append the log messages to the
       given file.  This is useful over ``file`` if multiple
       processes may need to write to the same log file.

     - ``stdout`` to send log messages to stdout.

   Multiple handlers may be specified by putting the whole value in
   quotes and separating the entries with a comma.

#. ``format``: Allows one to customize what each log message
   contains.  What this string may contain is described in the
   `logging module LogRecord Attributes
   <https://docs.python.org/3/library/logging.html#logrecord-attributes>`_
   section of the Python standard library.

Example
=======

The following configuration turns on all log messages and saves them
in the file myrun.log:

.. code-block:: ini

    [*]
    level = INFO
    handler = file:myrun.log
