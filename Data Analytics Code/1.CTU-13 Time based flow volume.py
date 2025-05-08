import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_ctu13_timebased_daily(
    input_csv: str = "cybersecurity_game/data/Malware Attack Patterns/Institutional/CTU-13 Botnet Dataset/CTU13_final_validated.csv",
    output_dir: str = "cybersecurity_game/docs/Data Analytics Results"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    df['StartTime'] = pd.to_datetime(df['StartTime'], unit='s', errors='coerce')
    df['TimeBin'] = df['StartTime'].dt.floor('1min')
    grouped = df.groupby(['TimeBin', 'label']).size().reset_index(name='FlowCount')
    pivoted = grouped.pivot(index='TimeBin', columns='label', values='FlowCount').fillna(0)

# FULL-RANGE PLOT
    plt.figure(figsize=(10, 4))
    if 0 in pivoted.columns:
        plt.plot(pivoted.index, pivoted[0], label='Benign', color='blue')
    if 1 in pivoted.columns:
        plt.plot(pivoted.index, pivoted[1], label='Malicious', color='red')

    plt.title('CTU13: Time-Based Flow Volume (Full Range)')
    plt.xlabel('Time (1-minute bins)')
    plt.ylabel('Number of Flows')
    plt.legend()
    plt.tight_layout()

    save_path_full = os.path.join(output_dir, "CTU13_TimeBasedFlowVolume_FullRange.png")
    plt.savefig(save_path_full, dpi=300)
    plt.close()
    print(f"Full-range plot saved to: {save_path_full}")

# DAILY PLOTS
    df['Date'] = df['StartTime'].dt.date
    unique_dates = sorted(df['Date'].unique()) 

    for d in unique_dates:
        day_df = df[df['Date'] == d].copy()

        if day_df.empty:
            continue

        day_df['TimeBin'] = day_df['StartTime'].dt.floor('1min')
        day_grouped = day_df.groupby(['TimeBin', 'label']).size().reset_index(name='FlowCount')
        day_pivoted = day_grouped.pivot(index='TimeBin', columns='label', values='FlowCount').fillna(0)

        plt.figure(figsize=(10, 4))
        if 0 in day_pivoted.columns:
            plt.plot(day_pivoted.index, day_pivoted[0], label='Benign', color='blue')
        if 1 in day_pivoted.columns:
            plt.plot(day_pivoted.index, day_pivoted[1], label='Malicious', color='red')

        plt.title(f"CTU13: Time-Based Flow Volume on {d}")
        plt.xlabel('Time (1-minute bins)')
        plt.ylabel('Number of Flows')
        plt.legend()
        plt.tight_layout()

        day_str = d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)
        save_path_day = os.path.join(output_dir, f"CTU13_TimeBasedFlowVolume_{day_str}.png")
        plt.savefig(save_path_day, dpi=300)
        plt.close()
        print(f"Daily plot for {d} saved to: {save_path_day}")

if __name__ == "__main__":
    analyze_ctu13_timebased_daily()


