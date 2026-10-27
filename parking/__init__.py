"""Parking space occupancy classification (PKLot -> CNRPark-EXT domain shift)."""

CLASSES = ["empty", "occupied"]
PATCH_SIZE = 64  # every slot is warped to a PATCH_SIZE x PATCH_SIZE RGB patch
