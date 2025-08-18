import streamlit as st
import pandas as pd
import numpy as np
import io
import re
from datetime import datetime

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

# --- Streamlit App Interface (Unchanged) ---
st.set_page_config(page_title="Donation Statistics Generator", layout="centered")
st.title("📊 Donation Statistics Generator")
st.write("Upload the three required Excel files to generate the performance report.")

donations_file = st.file_uploader("📁 1. Upload Donations File", type=["xlsx"])
names_file = st.file_uploader("📁 2. Upload Names File", type=["xlsx"])
workhours_file = st.file_uploader("📁 3. Upload Work Hours File", type=["xlsx"])

if st.button("🚀 Generate Report"):
    if donations_file and names_file and workhours_file:
        with st.spinner("Processing files... this may take a moment."):
            excel_data = process_files(donations_file, names_file, workhours_file)

        if excel_data:
            st.success("✅ Report generated successfully!")
            st.download_button(
                label="📥 Download Report",
                data=excel_data,
                file_name="worker_statistics_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("⚠️ Please upload all three files to generate the report.")
