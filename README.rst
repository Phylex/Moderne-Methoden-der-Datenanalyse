Moderne Methoden der Datenanalyse
=================================

Moderne Methoden der Datenanalyse ist eine Mastervorlesung am KIT die sich mit den aktuellen in Verwendung
befindlichen Methoden der Datenanalyse von Teilchenphysikalischen Daten beschaeftigt und Teile der Statistik
abdeckt. Es werden Methoden wie die des Maximum-Likelihood-Schaetzers oder die :math:`\chi^2` Methode Vorgestellt.

Des weiteren werden die Vorbedingungen und Annahmen besprochen die bei der Modellbildung eine Rolle spielen.

In den Vorlesungen kommen verschiedene Beispiele vor, die meistens mithilfe des Root_ Frameworks erstellt wurden.
Da ich die finde, dass die Root_ macros nicht sonderlich leserlich sind, habe ich beschlossen, zu meiner Uebung die meisten Beispiele in Python3_ nachimplementiere.

Bei der reimplementation werden die Packete numpy_ und matplotlib_ sowie seltener module aus den scipy_ packeten verwendet. Dies entspricht einem recht konservativem Scientific stack und sollte meistens recht gut installierbar sein

Ausfuehren der Scripte
----------------------
Die meisten Scripte sind ausfuehrbar, unter der Bedingung, dass sie auf einem Unixoiden OS ausgefuehrt werden und
sich eine Python instanz unter :code:`/usr/lib/python3` befindet fuer welche die noetigen Packete installiert wurden.

.. _Pyhton3: https://www.python.org
.. _numpy: https://numpy.org
.. _matplotlib: https://matplotlib.org
.. _Root: https://root.cern.ch
