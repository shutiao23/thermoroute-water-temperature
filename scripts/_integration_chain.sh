#!/usr/bin/env bash
cd "/Users/liziqing/Desktop/Part-time_Work/[副业]论文/0625_环境工程/project1"
export OMP_NUM_THREADS=8 PYTHONPATH=src
LOG=outputs/logs/integration.log
echo "[$(date +%H:%M)] chain start" > $LOG
# 1. wait air2stream splice
until grep -q "DONE" outputs/logs/rerun_air2stream.log 2>/dev/null; do sleep 30; done
echo "[$(date +%H:%M)] air2stream done" >> $LOG
# 2. per-station LightGBM splice (M4)
python3 -u scripts/_perstation_lgb.py >> $LOG 2>&1
echo "[$(date +%H:%M)] per-station LGB spliced" >> $LOG
# 3. wait region-transfer folds (4 checkpoints)
until [ -f outputs/predictions/region_ckpt/tr_fold0.parquet ] && \
      [ -f outputs/predictions/region_ckpt/tr_fold1.parquet ] && \
      [ -f outputs/predictions/region_ckpt/tr_fold2.parquet ] && \
      [ -f outputs/predictions/region_ckpt/tr_fold3.parquet ]; do sleep 60; done
echo "[$(date +%H:%M)] region folds done" >> $LOG
# 4. region-transfer assemble (B1+B2 verdict)
python3 -u scripts/13c_region_transfer.py --assemble >> $LOG 2>&1
echo "[$(date +%H:%M)] region assemble done" >> $LOG
# 5. re-run downstream on updated v2
python3 -u scripts/10_usgs_analysis.py >> $LOG 2>&1; echo "[$(date +%H:%M)] stage10 done" >> $LOG
python3 -u scripts/12_claim_stats.py >> $LOG 2>&1; echo "[$(date +%H:%M)] stage12 done" >> $LOG
python3 -u scripts/13_rigor.py >> $LOG 2>&1; echo "[$(date +%H:%M)] stage13 done" >> $LOG
python3 -u scripts/15_stratified.py >> $LOG 2>&1; echo "[$(date +%H:%M)] stage15 done" >> $LOG
echo "[$(date +%H:%M)] CHAIN COMPLETE" >> $LOG
