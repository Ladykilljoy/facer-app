import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

# Try to import lifelines; show a friendly note if it's missing.
try:
    from lifelines import KaplanMeierFitter
    LIFELINES_AVAILABLE = True
except Exception:
    LIFELINES_AVAILABLE = False

# Helper function to parse hours from the schedule's time column
def get_shift_hours(time_str):
    if not isinstance(time_str, str):
        return 4 # Default to 4 if data is missing/malformed

    match_explicit = re.search(r'(\d+)\s*óra', time_str)
    if match_explicit:
        return int(match_explicit.group(1))

    match_range = re.search(r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', time_str)
    if match_range:
        try:
            time_format = "%H:%M"
            start_time = datetime.strptime(match_range.group(1), time_format)
            end_time = datetime.strptime(match_range.group(2), time_format)
            duration = end_time - start_time
            return duration.total_seconds() / 3600
        except ValueError:
            return 4

    return 4

# Main function to process the dataframes
def process_files(donations_file, names_file, workhours_file):
    try:
        # Load data
        donations_df = pd.read_excel(donations_file, skiprows=14)
        names_df = pd.read_excel(names_file)
        workhours_df = pd.read_excel(workhours_file, header=None)

        # --- Sheet 1: Main Performance Report Logic (Unchanged) ---
        donations_df.columns = donations_df.columns.str.strip()
        donations_df = donations_df.iloc[:-3]
        donations_df['Amount'] = pd.to_numeric(donations_df['Amount'], errors='coerce')
        donations_sum = donations_df.groupby('Campaign: Campaign Name')['Amount'].sum().reset_index()
        donations_sum.rename(columns={'Campaign: Campaign Name': 'Campaign name', 'Amount': 'Total Donation Amount'}, inplace=True)
        donations_count = donations_df.groupby('Campaign: Campaign Name').size().reset_index(name='Number of Donations')
        donations_count.rename(columns={'Campaign: Campaign Name': 'Campaign name'}, inplace=True)

        hours_per_person = {}
        for index, row in workhours_df.iterrows():
            time_str = row[3]
            if pd.isna(time_str): continue
            shift_hours = get_shift_hours(time_str)
            for col in [4, 5, 6]:
                if pd.notna(row[col]):
                    name = str(row[col]).strip()
                    if name and name not in ['---', 'nan']:
                        hours_per_person[name] = hours_per_person.get(name, 0) + shift_hours
        hours_df = pd.DataFrame(list(hours_per_person.items()), columns=['Név', 'Total Hours'])

        final_df = pd.merge(names_df, hours_df, on='Név', how='left')
        final_df = pd.merge(final_df, donations_sum, on='Campaign name', how='left')
        final_df = pd.merge(final_df, donations_count, on='Campaign name', how='left')
        final_df.fillna(0, inplace=True)
        final_df['Total Wage'] = final_df['Bér'] * final_df['Total Hours']
        final_df['Donations (Count) / Hour'] = np.divide(final_df['Number of Donations'], final_df['Total Hours'])
        final_df['ROI'] = np.divide((final_df['Total Donation Amount'] - final_df['Total Wage']), final_df['Total Wage'])
        final_df.fillna(0, inplace=True)
        final_df.replace([np.inf, -np.inf], 0, inplace=True)

        # --- Sheet 2: Worker Calendar Logic (NEW RUNNING COUNT) ---
        
        # 1. Get clean work and donation data
        workhours_df[0] = workhours_df[0].ffill()
        all_shifts = []
        for index, row in workhours_df.iterrows():
            date = row[0]
            if pd.notna(date) and pd.notna(row[3]):
                for col in [4, 5, 6]:
                    if pd.notna(row[col]):
                        name = str(row[col]).strip()
                        if name and name not in ['---', 'nan']:
                            all_shifts.append({'Név': name, 'Date': date})
        all_shifts_df = pd.DataFrame(all_shifts).drop_duplicates()
        all_shifts_df['Date'] = pd.to_datetime(all_shifts_df['Date'])
        
        donations_df['Transaction Date'] = pd.to_datetime(donations_df['Transaction Date'], dayfirst=True)
        donations_with_names = pd.merge(donations_df, names_df, left_on='Campaign: Campaign Name', right_on='Campaign name')
        donation_dates_by_worker = donations_with_names.groupby('Név')['Transaction Date'].apply(lambda x: set(x.dt.date))
        
        # 2. Iteratively build the calendar with the running count
        calendar_data = {}
        all_workers_in_schedule = sorted(all_shifts_df['Név'].unique())
        
        if not all_shifts_df.empty:
            month_start = all_shifts_df['Date'].min().replace(day=1)
            days_in_month = pd.Period(month_start, 'M').days_in_month
            month_dates = [month_start + pd.Timedelta(days=i) for i in range(days_in_month)]

            for worker in all_workers_in_schedule:
                worker_work_dates = set(all_shifts_df[all_shifts_df['Név'] == worker]['Date'].dt.date)
                worker_donation_dates = donation_dates_by_worker.get(worker, set())
                
                worked_row = {}
                donated_row = {}
                consecutive_zeros_count = 0

                for date in month_dates:
                    day = date.day
                    had_donation_today = date.date() in worker_donation_dates
                    had_shift_today = date.date() in worker_work_dates

                    # Any donation (even on non-workdays) resets the counter
                    if had_donation_today:
                        consecutive_zeros_count = 0
                    
                    # Populate the "Donated" row
                    donated_row[day] = '✓' if had_donation_today else ''
                    
                    # Populate the "Worked" row with the running count logic
                    if had_shift_today:
                        if had_donation_today:
                            worked_row[day] = 0
                        else:
                            consecutive_zeros_count += 1
                            worked_row[day] = consecutive_zeros_count
                    else:
                        worked_row[day] = ''

                calendar_data[(worker, 'Worked')] = worked_row
                calendar_data[(worker, 'Donated')] = donated_row

        # 3. Create DataFrame from the generated data
        worker_calendar = pd.DataFrame.from_dict(calendar_data, orient='index')
        if not worker_calendar.empty:
             worker_calendar.index = pd.MultiIndex.from_tuples(worker_calendar.index, names=['Név', 'Metric'])
             worker_calendar.sort_index(inplace=True)
             worker_calendar = worker_calendar.reindex(columns=range(1, days_in_month + 1), fill_value='')

        # --- Final Output Generation ---
        output_df = final_df[['Név', 'Total Hours', 'Number of Donations', 'Total Donation Amount', 'Total Wage', 'Donations (Count) / Hour', 'ROI']]
        output_df.rename(columns={'Név': 'Worker Name', 'Total Donation Amount': 'Total Donations (HUF)'}, inplace=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='Worker Performance')
            worker_calendar.to_excel(writer, index=True, sheet_name='Worker Calendar')
        
        processed_data = output.getvalue()
        return processed_data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure the uploaded files are in the correct format and all columns are present.")
        return None


# ------------------------------
# New functionality: Kaplan–Meier survival
# ------------------------------

def compute_km_from_transactions(df, id_col, date_col, status_col, cutoff_str, grace_days, dayfirst):
    # Parse dates
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    df = df.dropna(subset=[date_col])

    # We assume each row is a successful recurring transaction.
    # Aggregate per donor
    g = df.groupby(id_col, as_index=False).agg(
        first_donation=(date_col, "min"),
        last_donation=(date_col, "max"),
        donations_count=(date_col, "count")
    )

    cutoff = pd.to_datetime(cutoff_str)

    # Determine event (churn) vs censored
    if status_col and status_col in df.columns:
        last_rows = df.sort_values(date_col).groupby(id_col).tail(1)
        status_map = last_rows.set_index(id_col)[status_col]
        g["status"] = g[id_col].map(status_map)
        # Treat anything not equal to "Active" as churned
        g["event_observed"] = (g["status"].fillna("Active") != "Active").astype(int)
        g["event_time"] = g.apply(
            lambda r: (min(r["last_donation"], cutoff) - r["first_donation"]).days
            if r["event_observed"] == 0
            else (r["last_donation"] - r["first_donation"]).days,
            axis=1
        )
    else:
        grace_cut = cutoff - pd.Timedelta(days=grace_days)
        g["event_observed"] = (g["last_donation"] <= grace_cut).astype(int)
        g["event_time"] = g.apply(
            lambda r: (cutoff - r["first_donation"]).days
            if r["event_observed"] == 0
            else (r["last_donation"] - r["first_donation"]).days,
            axis=1
        )

    g = g[g["event_time"] >= 0].copy()

    # Fit KM
    kmf = KaplanMeierFitter()
    kmf.fit(durations=g["event_time"], event_observed=g["event_observed"], label="Donor survival")

    # Survival checkpoints
    checkpoints = {
        "3 months": 91,
        "6 months": 182,
        "12 months": 365
    }
    surv_points = {k: float(kmf.survival_function_at_times(v).iloc[0]) for k, v in checkpoints.items()}

    return g, kmf, surv_points

# --- Streamlit App Interface ---
st.set_page_config(page_title="Donation Statistics Generator", layout="centered")
st.title("📊 Donation Statistics Suite")

tab1, tab2 = st.tabs(["Performance Report", "Donor Survival (Kaplan–Meier)"])

# ---------------- Tab 1: original app ----------------
with tab1:
    st.header("Worker Performance Report")
    st.write("Upload the three required Excel files to generate the performance report.")

    donations_file = st.file_uploader("📁 1. Upload Donations File", type=["xlsx"], key="donations_file_perf")
    names_file = st.file_uploader("📁 2. Upload Names File", type=["xlsx"], key="names_file_perf")
    workhours_file = st.file_uploader("📁 3. Upload Work Hours File", type=["xlsx"], key="workhours_file_perf")

    if st.button("🚀 Generate Report", key="run_perf"):
        if donations_file and names_file and workhours_file:
            st.info("Processing files... please wait.")
            excel_data = process_files(donations_file, names_file, workhours_file)
            if excel_data:
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label="📥 Download Report",
                    data=excel_data,
                    file_name="worker_statistics_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_perf"
                )
        else:
            st.warning("⚠️ Please upload all three files to generate the report.")

# ---------------- Tab 2: Kaplan–Meier ----------------
with tab2:
    st.header("Donor Survival (Kaplan–Meier)")
    st.caption(
        "Estimate how long recurring donors keep donating. Handles active donors via censoring. "
        "Outputs median survival and a survival curve."
    )

    if not LIFELINES_AVAILABLE:
        st.error("The 'lifelines' package is not installed. Add `lifelines` to your requirements.txt and redeploy.")
    else:
        donations_file_km = st.file_uploader("📁 Upload Recurring Transactions (Excel/CSV)", type=["xlsx", "csv"], key="donations_file_km")

        with st.expander("⚙️ Columns & Settings"):
            id_col = st.text_input("Donor ID column", value="(T)ID")
            date_col = st.text_input("Transaction date column", value="Transaction Date")
            status_col = st.text_input("Optional status column (leave blank if none)", value="Status")
            cutoff = st.text_input("Dataset cutoff date (YYYY-MM-DD)", value="2025-08-22")
            grace_days = st.number_input("Grace days (used if no status column)", min_value=0, max_value=180, value=45, step=1)
            dayfirst = st.checkbox("Dates are day-first (e.g., 25/06/2025)", value=True)
            min_don = st.number_input("Minimum transactions to include", min_value=1, max_value=20, value=1, step=1)

        run_km = st.button("📈 Run Kaplan–Meier", key="run_km")

        if run_km:
            if donations_file_km is None:
                st.warning("⚠️ Please upload the transactions file.")
            else:
                # Load file
                if donations_file_km.name.lower().endswith((".xls", ".xlsx", ".xlsm", ".xlsb")):
                    df = pd.read_excel(donations_file_km)
                else:
                    df = pd.read_csv(donations_file_km)

                # basic column existence check
                missing = [c for c in [id_col, date_col] if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    # Filter by min donations per donor
                    counts = df.groupby(id_col)[date_col].count().rename("cnt")
                    valid_ids = counts[counts >= min_don].index
                    df = df[df[id_col].isin(valid_ids)].copy()

                    # If status_col is empty or not present, treat it as None
                    status_col_in = status_col.strip() if status_col.strip() and status_col in df.columns else None

                    try:
                        g, kmf, surv_points = compute_km_from_transactions(
                            df, id_col=id_col, date_col=date_col, status_col=status_col_in,
                            cutoff_str=cutoff, grace_days=int(grace_days), dayfirst=bool(dayfirst)
                        )

                        # Summary
                        st.subheader("Summary")
                        events = int(g["event_observed"].sum())
                        censored = int((1 - g["event_observed"]).sum())
                        median_days = kmf.median_survival_time_
                        median_months = median_days / 30.4375 if pd.notna(median_days) else None

                        st.write(f"Donors included: **{len(g)}**")
                        st.write(f"Events (churn): **{events}**, Censored (still active): **{censored}**")
                        if pd.notna(median_days):
                            st.write(f"Median survival: **{median_days:.1f} days** (~{median_months:.2f} months)")
                        else:
                            st.write("Median survival: not reached (curve never drops below 0.5).")

                        st.write("Survival probabilities:")
                        for label, val in surv_points.items():
                            st.write(f"• S({label}) ≈ **{val:.3f}**")

                        # Plot
                        fig, ax = plt.subplots()
                        kmf.plot(ci_show=True, ax=ax)
                        ax.set_xlabel("Days since first donation")
                        ax.set_ylabel("Survival probability (still donating)")
                        ax.set_title("Kaplan–Meier Donor Survival")
                        st.pyplot(fig, clear_figure=True)

                        # Downloads
                        out = g[[id_col, "first_donation", "last_donation", "donations_count", "event_time", "event_observed"]].copy()
                        out.rename(columns={"event_time": "duration_days", "event_observed": "event"}, inplace=True)

                        # Per-donor CSV
                        buf1 = io.BytesIO()
                        out.to_csv(buf1, index=False)
                        st.download_button("📥 Download per-donor durations (CSV)", buf1.getvalue(), file_name="donor_durations.csv", mime="text/csv", key="dl_durations")

                        # KM survival function CSV
                        buf2 = io.BytesIO()
                        kmf.survival_function_.to_csv(buf2)
                        st.download_button("📥 Download survival function (CSV)", buf2.getvalue(), file_name="km_survival_function.csv", mime="text/csv", key="dl_curve")

                    except Exception as e:
                        st.exception(e)

