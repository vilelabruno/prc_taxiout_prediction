TARGET = "TAXITIME_SEC_mvt"
ID_COL = "MVT_ID_mvt"
PHASE_COL = "PHASE_mvt"

TIMESTAMP_COLUMNS = (
    "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
    "LOBT_flt", "IOBT_flt", "EOBT_1_flt", "ARVT_1_flt", "AOBT_3_flt", "ARVT_3_flt",
)

TRAINING_MONTHS = tuple(f"2025-{m:02d}" for m in range(1, 13))
DEFAULT_HOLDOUT_MONTHS = (1, 7)


def training_filename(month):
    year, mon = (int(x) for x in month.split("-"))
    nyear, nmon = (year + 1, 1) if mon == 12 else (year, mon + 1)
    return f"training_{year}-{mon:02d}-01_{nyear}-{nmon:02d}-01.parquet"
