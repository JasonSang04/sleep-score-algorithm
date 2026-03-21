import pandas as pd
import numpy as np

# Data pre-processing
df = pd.read_csv("whoop_fitness_dataset_100k.csv")
col_names = ["user_id","date","recovery_score","sleep_hours","sleep_efficiency","rem_sleep_hours","deep_sleep_hours","wake_ups",
             "time_to_fall_asleep_min","hrv","resting_heart_rate","hrv_baseline","rhr_baseline","respiratory_rate"]
df_clean = df[col_names].copy()

# Score dimension 1: Sleep duration and efficiency
def score1(row):
    
    # Calculate sleep duration score (asymmetric Gaussian)
    def duration_score(sleep_hours):
        target_hours = 7.5
        sigma_low = 1.2
        sigma_high = 2
        if sleep_hours < target_hours:
            sigma = sigma_low
        else:
            sigma = sigma_high
        score = 100 * np.exp(-((sleep_hours - target_hours) ** 2) / (2 * sigma ** 2))
        return max(0, min(100, score))
    
    # Calculate sleep efficiency score (piecewise linear function with threshold saturation)
    def efficiency_score(sleep_efficiency):
        if sleep_efficiency >= 90:
            return 100
        elif sleep_efficiency >= 85:
            return 80 + (sleep_efficiency - 85) / 5 * 20
        elif sleep_efficiency >= 70:
            return (sleep_efficiency - 70) / 15 * 80
        else:
            return 0
    
    sleep_hours = row["sleep_hours"]
    sleep_efficiency = row["sleep_efficiency"]
    
    return duration_score(sleep_hours) * 0.6 + efficiency_score(sleep_efficiency) * 0.4

# Score dimension 2: Sleep architecture
def score2(row):
    # Calculate deep and REM sleep scores (piecewise linear function)
    sleep_hours = row["sleep_hours"]
    deep_hours = row["deep_sleep_hours"]
    rem_hours = row["rem_sleep_hours"]
    
    if sleep_hours > 0:
        deep_percentage = deep_hours / sleep_hours
        rem_percentage = rem_hours / sleep_hours
    else:
        return 0
    
    target_deep = 0.18
    target_rem = 0.23
    tolerance = 0.08
    
    deep_score = 100 * max(0, 1 - abs(deep_percentage - target_deep) / tolerance)
    rem_score = 100 * max(0, 1 - abs(rem_percentage - target_rem) / tolerance)
    
    return deep_score * 0.6 + rem_score * 0.4

# Score dimension 3: Physiological recovery
def score3(row):
    # Calculate HRV score (sigmoid)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    hrv = row["hrv"]
    hrv_base = row["hrv_baseline"]
    hrv_ratio = hrv / hrv_base
    
    k_hrv = 4.62
    x0_hrv = 0.7
    x = k_hrv * (hrv_ratio - x0_hrv)
    hrv_score = 100 * sigmoid(x)
    
    # Calculate heart rate score (asymmetric Gaussian)
    rhr = row["resting_heart_rate"]
    rhr_base = row["rhr_baseline"]
    rhr_diff = rhr - rhr_base
    
    sigma_low = 25
    sigma_high = 10
    if rhr_diff < 0:
        sigma = sigma_low
    else:
        sigma = sigma_high
        
    rhr_score = 100 * np.exp(-0.5 * (rhr_diff / sigma) ** 2)
    rhr_score = max(0, min(100, rhr_score))    
    
    # Calculate respiratory rate score (symmetric Gaussian)
    rr = row["respiratory_rate"]
    target_rr = 16
    diff_rr = abs(rr - target_rr)
    sigma_rr = 4
    rr_score = 100 * np.exp(-0.5 * (diff_rr / sigma_rr) ** 2)
    rr_score = max(0, min(100, rr_score))
    
    return hrv_score * 0.6 + rhr_score * 0.25 + rr_score * 0.15
    
# Score dimension 4: Sleep latency and continuity
def score4(row):
    # Calculate sleep latency score (tiered scoring with optimal range)
    latency_min = row["time_to_fall_asleep_min"]
    if latency_min <= 15:
        if latency_min < 5:
            latency_score = 90 + (latency_min / 5) * 10
        else:
            latency_score = 100
    else:
        latency_score = max(0, 100 - (latency_min - 15) * 2.22)

    # Calculate sleep continuity score (linear penalty)
    wakeups = row["wake_ups"]
    continuity_score = 100 * max(0, 1 - (wakeups / 5))
    
    return latency_score * 0.4 + continuity_score * 0.6

# The main function
def calculate_sleep_score(row):
    return score1(row) * 0.3 + score2(row) * 0.25 + score3(row) * 0.3 + score4(row) * 0.15

print("Calculating scores...")
df_clean["final_sleep_score"] = df_clean.apply(calculate_sleep_score, axis=1)
df_clean["final_sleep_score"] = df_clean["final_sleep_score"].round(0).astype(int)

print("\nScore Statistics:")
print(df_clean["final_sleep_score"].describe())


# Validate the range of final sleep scores
if df_clean["final_sleep_score"].max() > 100.01 or df_clean["final_sleep_score"].min() < -0.01:
    print("Warning: Scores out of range [0, 100] detected!")
else:
    print("All scores are within the valid range [0, 100].")


# Explore the correlation between recovery scores and sleep scores
print("="*60)
df_corr_analysis = df_clean[["recovery_score","final_sleep_score"]].copy()
corr_matrix = df_corr_analysis.corr(method="pearson")
corr_recovery = corr_matrix.loc["recovery_score","final_sleep_score"]
print(f"Correlation (recovery score vs. sleep score): {corr_recovery:.4f}")

# Advanced correlation analysis: Explore the correlation between daily recovery score changes and sleep scores
df_corr_advanced = df_clean[["user_id","date","recovery_score","final_sleep_score"]].copy()
df_corr_advanced["date"] = pd.to_datetime(df_corr_advanced["date"])
df_corr_advanced = df_corr_advanced.sort_values(by=["user_id","date"]).reset_index(drop=True)
df_corr_advanced["recovery_diff"] = df_corr_advanced.groupby("user_id")["recovery_score"].diff()
df_corr_adv_clean = df_corr_advanced.dropna(subset=["recovery_diff","final_sleep_score"]).copy()

score_global = df_corr_adv_clean["recovery_diff"].corr(df_corr_adv_clean["final_sleep_score"])
user_corrs = df_corr_adv_clean.groupby("user_id")[["recovery_diff","final_sleep_score"]].apply(
    lambda x: x["recovery_diff"].corr(x["final_sleep_score"])).dropna()
score_avg_user = user_corrs.mean()

print(f"Global corr: {score_global:.4f}")
print(f"Average user corr: {score_avg_user:.4f}")


# Export the results to .csv file
output_filename = "results.csv"
df_clean.to_csv(output_filename, index=False)

