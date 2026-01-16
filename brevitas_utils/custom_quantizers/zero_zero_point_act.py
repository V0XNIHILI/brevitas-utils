from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector


class ZeroZeroPointAct(ExtendedInjector):
    zero_point_impl = ZeroZeroPoint
