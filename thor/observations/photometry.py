import quivr as qv

__all__ = ["Photometry"]


class Photometry(qv.Table):

    mag = qv.Float64Column()
    mag_sigma = qv.Float64Column(nullable=True)
    filter = qv.LargeStringColumn()
