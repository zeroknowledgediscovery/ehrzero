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

