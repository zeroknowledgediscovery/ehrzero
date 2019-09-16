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

    from ehrzero import ehrzero as ehr
    risks = ehr.predict_with_confidence(source_filepath,out_filepath,
                                        separator=',',delimitor=' ',
					n_first_weeks=[week1,week2,..,weekn])
					

**Installation**

.. code-block::
   
   pip3 install ehrzero --user --upgrade

**EHR data format**

Diagnostic data stored in text file, one line per patient as follows: patient id, gender and list of space-separated, comma-delineated diagnosis records, all separated by spaces. Each diagnosis record consists of the week since the start of observation, followed by comma and the ICD-9 code of the diagnosis. 

Example of a diagnostic record for Lorax, male:

.. code-block::

   Lorax,M 5,277.03 10,611.79 18,057.8 58,157.8 78,057.8 108,057.8 128,057.8 148,057.8
   <patient_id,gender> <code,week> <code,week> ...

   
**Risk Estimation Example As Python Script**

With properly formatted  patient diagnostic history, we use  function *predict_with_confidence* and specify the data filepath  and  the weeks of age at which  we want to estimate risk. Optionally, we  may also specify the separator (separating code and week, see example above) and delimitor (separating two code-week pairs, see example above)  for the diganostic records within file (space and comma are default).

The predict_with_confidence function returns the predicted risk of autism for each patient for each requested week.

.. code-block::

   from ehrzero import ehrzero as ehr
   source = './src.dat'
   outfile = './out.dat'
   weeks = [100,200]
    
   risks = ehr.predict_with_confidence(source,outfile,
                                       separator=',',delimitor=' ',
				       n_first_weeks=weeks)
					


				       
**Command Line Execution**


A commandline tool *zero.py* is included in the package.

Invoking the tool with examples of diagnostic history
included in the package may be done as follows:

.. code-block::

   python zero.py -data ehrzero/example/tests/exD1.dx  -n 100 200 300 

   patient_id week risk relative_risk confidence
   A032532061 150 0.019578 1.126543 98.86
   A032532061 175 0.058364 3.358280 99.93
   A032532061 200 0.031544 1.815065 99.56

