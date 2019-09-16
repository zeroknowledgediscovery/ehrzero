===============
ehrzero
===============

.. figure:: https://img.shields.io/pypi/dm/ehrzero.svg?style=plastic
   :alt: ehrzero PyPI Downloads
.. figure:: https://img.shields.io/pypi/v/ehrzero.svg
   :alt: ehrzero version

	 

.. image:: http://zed.uchicago.edu/logo/logozed1.png
   :height: 400px
   :scale: 50 %
   :alt: alternate text
   :align: center


.. class:: no-web no-pdf

:Info: Zero-Knowledge Risk Oracle
:Author: ZeD@UChicago <zed.uchicago.edu>
:Description: Estimation of the risk of future diagnoses of
	      neuropsychiatric disorders (particularly autism) in early childhood,
	      based on the diagnostic codes recorded during
	      doctor visits. The prediction pipeline is based on
	      inferring optimal stochastic generators for diagnostic code sequences,
	      and detecting subtle deviations that drive up risk of
	      an eventual neuropsychiatric diagnoses. The out-of-sample
	      AUC score on the Truven dataset of insurance claims
	      (close to 3 million children in out-of-sample data) is just over 80%,
	      for both males and females.


**Usage:**

.. code-block::

    from ehrzero import ehrzero
    ehr.predict_with_confidence(SOURCE,n_first_weeks)

**Installation**

.. code-block::
   pip3 install ehrzero --user --upgrade

**EHR data format**

Diagnostic data stored in text file, one line per patient as follows: patient id, gender and list of space-separated, comma-delineated diagnosis records, all separated by spaces. Each diagnosis record consists of the week since the start of observation, followed by comma and the ICD-9 code of the diagnosis. 

Example of a patient line:

.. code-block::

   Lorax,M 5,277.03 10,611.79 18,057.8 58,157.8 78,057.8 108,057.8 128,057.8 148,057.8
