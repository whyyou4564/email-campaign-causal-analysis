"""阶段一、二：可验证的数据获取，以及不改变实验样本的数据审计。"""
import gzip
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd
import requests

from .config import RAW, OUT, PRE_COLUMNS, ARMS, OUTCOMES, ensure_directories

SOURCE = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
SOURCE_MD5 = "a68a81291f53a14f4e29002629803ba3"
LOCAL_SHA256 = "00a6a868e05a9ffe7382da51629f6d6dce88c5acfc945e79d314ebc78fd3a2c0"

def download():
    """缓存存在时只验证；缺失时从官方文档所列镜像下载。"""
    ensure_directories()
    csv_path = RAW / "hillstrom.csv"
    archive_path = RAW / "hillstrom.csv.gz"
    if csv_path.exists() and not archive_path.exists():
        # 本次来自用户已有 ZIP 中的公开原始 CSV，未复制其中的分析代码。
        blob = csv_path.read_bytes()
        if hashlib.sha256(blob).hexdigest() != LOCAL_SHA256:
            raise ValueError("本地 CSV 与本项目已核对的数据校验值不一致。")
        acquisition = "local public CSV; extracted from user's existing dataset archive"
        source_checksum_verified = False
    else:
        if not archive_path.exists():
            response = requests.get(SOURCE, timeout=120)
            response.raise_for_status()
            archive_path.write_bytes(response.content)
        compressed = archive_path.read_bytes()
        if hashlib.md5(compressed).hexdigest() != SOURCE_MD5:
            raise ValueError("下载文件与官方文档公布的 MD5 不一致。")
        blob = gzip.decompress(compressed)
        if not csv_path.exists() or csv_path.read_bytes() != blob:
            csv_path.write_bytes(blob)
        acquisition = "official scikit-uplift documented mirror"
        source_checksum_verified = True

    metadata = {
        "documented_download_source": SOURCE,
        "expected_archive_md5": SOURCE_MD5,
        "archive_md5_verified_this_run": source_checksum_verified,
        "csv_sha256": hashlib.sha256(blob).hexdigest(),
        "acquisition": acquisition,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_source": "https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
    }
    (OUT / "data_provenance.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return load()

def load():
    return pd.read_csv(RAW / "hillstrom.csv")

def audit(df):
    """检查字段、缺失、重复、范围和 visit/conversion/spend 的逻辑。"""
    ensure_directories()
    expected = set(PRE_COLUMNS + ["segment"] + OUTCOMES)
    if set(df.columns) != expected:
        raise ValueError(f"字段发生变化：{set(df.columns) ^ expected}")
    if set(df.segment) != set(ARMS):
        raise ValueError("未知实验分组。")

    pd.DataFrame({
        "field": df.columns,
        "dtype": df.dtypes.astype(str).values,
        "missing": df.isna().sum().values,
        "unique": df.nunique().values,
    }).to_csv(OUT / "schema_audit.csv", index=False)
    df.select_dtypes("number").describe(
        percentiles=[.01, .5, .99]
    ).T.to_csv(OUT / "numeric_audit.csv")
    df.segment.value_counts().reindex(ARMS).rename("n").to_csv(OUT / "arm_counts.csv")

    findings = {
        "rows": len(df), "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "identical_rows": int(df.duplicated().sum()),
        "duplicate_pre_feature_rows": int(df.duplicated(PRE_COLUMNS).sum()),
        "no_customer_id": True,
        "recency_out_of_range": int((~df.recency.between(1, 12)).sum()),
        "negative_history_or_spend": int(((df.history < 0) | (df.spend < 0)).sum()),
        "non_binary": int(sum(
            (~df[column].isin([0, 1])).sum()
            for column in ["mens", "womens", "newbie", "visit", "conversion"]
        )),
        "purchase_without_visit": int(((df.conversion == 1) & (df.visit == 0)).sum()),
        "spend_conversion_inconsistent": int(((df.spend > 0) != (df.conversion == 1)).sum()),
        "spend_max": float(df.spend.max()),
        "spend_zero_share": float((df.spend == 0).mean()),
        "history_iqr_high_count": int((
            df.history > df.history.quantile(.75)
            + 1.5 * (df.history.quantile(.75) - df.history.quantile(.25))
        ).sum()),
        "action": "Retain identical profiles and valid high spending; no deduplication or outcome trimming.",
    }
    (OUT / "audit.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    fatal = [
        "missing_cells", "recency_out_of_range", "negative_history_or_spend",
        "non_binary", "purchase_without_visit", "spend_conversion_inconsistent",
    ]
    if any(findings[name] for name in fatal):
        raise ValueError(f"质量检查失败：{findings}")
    return findings
