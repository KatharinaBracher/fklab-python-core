"""
=======================
Probes (:mod:`fklab.io`)
=======================

.. currentmodule:: fklab.io

class template for development of future probes class

"""


__all__ = ["template_probe"]


class template_probe:
    def __init__(self):
        self._xCoord = None
        self._yCoord = None
        self._elecInd = None
        self._connected = None
        self._chanMap = None
        self._chanMap0ind = None
        self._kCoord = None

    @property
    def yCoord(self):

        if not self._yCoord:
            self.create_channel_map()

        return self._yCoord

    @property
    def xCoord(self):

        if not self._xCoord:
            self.create_channel_map()

        return self._xCoord

    @property
    def kCoord(self):

        if not self._kCoord:
            self.create_channel_map()

        return self._kCoord

    @property
    def elecInd(self):

        if not self._elecInd:
            self.create_channel_map()

        return self._elecInd

    @property
    def connected(self):
        if not self._connected:
            self.create_channel_map()

        return self._connected

    @property
    def chanMap(self):
        if not self._chanMap:
            self.create_channel_map()
        return self._chanMap

    @property
    def chanMap0ind(self):

        if not self._chanMap0ind:
            self.create_channel_map()

        return self._chanMap0ind

    def create_channel_map(self):
        print("not implemented at this level")
        pass
