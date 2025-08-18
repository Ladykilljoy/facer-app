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

    # Case 1: Explicit hours given (e.g., "-> 2 óra")
    match_explicit = re.search(r'(\d+)\s*óra', time_str)
    if match_explicit:
        return int(match_explicit.group(1))

    # Case 2: Standard time range (e.g., "10:00-14:00")
    match_range = re.search(r'(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})', time_str)
    if match_range:
        try:
            time_format = "%H:%M"
            start_time = datetime.strptime(match_range.group(1), time_format)
            end_time = datetime.strptime(match_range.group(2), time_format)
            duration = end_time - start_time
            # Return duration in hours
            return duration.total_seconds() / 3600
        except ValueError:
            # If time parsing fails, fall back to default
            return 4

    # Default case if no pattern matches
    return 4

# Main function to process the dataframes
def process_files(donations_file, names_file, workhours_file):
    try:
        donations_df = pd.read_excel(donations_file, skiprows=14)
        names_df = pd.read_excel(names_file)
        workhours_df = pd.read_excel(workhours_file, header=None)

        # --- Clean and Process Donations Data ---
        donations_df.columns = donations_df.columns.str.strip()
        donations_df = donations_df.iloc[:-3]
        donations_df['Amount'] = pd.to_numeric(donations_df['Amount'], errors='coerce')

        donations_sum = donations_df.groupby('Campaign: Campaign Name')['Amount'].sum().reset_index()
        donations_sum.rename(columns={'Campaign: Campaign Name': 'Campaign name', 'Amount': 'Total Donation Amount'}, inplace=True)

        donations_count = donations_df.groupby('Campaign: Campaign Name').size().reset_index(name='Number of Donations')
        donations_count.rename(columns={'Campaign: Campaign Name': 'Campaign name'}, inplace=True)

        # --- NEW: Calculate Total Hours Worked Dynamically ---
        hours_per_person = {}
        name_columns = [4, 5, 6]
        time_column = 3

        for index, row in workhours_df.iterrows():
            time_str = row[time_column]
            # Skip rows that don't have a time entry (likely empty rows)
            if pd.isna(time_str):
                continue
            
            shift_hours = get_shift_hours(time_str)

            for col in name_columns:
                # Ensure the cell contains a valid name before processing
                if pd.notna(row[col]):
                    name = str(row[col]).strip()
                    if name and name not in ['---', 'nan']:
                        hours_per_person[name] = hours_per_person.get(name, 0) + shift_hours
        
        hours_df = pd.DataFrame(list(hours_per_person.items()), columns=['Név', 'Total Hours'])

        # --- Merge DataFrames and Calculate All Metrics ---
        final_df = pd.merge(names_df, hours_df, on='Név', how='left')
        final_df = pd.merge(final_df, donations_sum, on='Campaign name', how='left')
        final_df = pd.merge(final_df, donations_count, on='Campaign name', how='left')

        final_df['Total Hours'].fillna(0, inplace=True)
        final_df['Total Donation Amount'].fillna(0, inplace=True)
        final_df['Number of Donations'].fillna(0, inplace=True)

        final_df['Total Wage'] = final_df['Bér'] * final_df['Total Hours']
        final_df['Donations (Count) / Hour'] = np.divide(final_df['Number of Donations'], final_df['Total Hours'])
        final_df['Donations (Count) / Hour'].fillna(0, inplace=True)
        final_df['ROI'] = np.divide((final_df['Total Donation Amount'] - final_df['Total Wage']), final_df['Total Wage'])
        final_df.loc[final_df['Total Wage'] == 0, 'ROI'] = np.inf
        final_df.loc[(final_df['Total Wage'] == 0) & (final_df['Total Donation Amount'] == 0), 'ROI'] = 0

        # --- Format and Prepare Output ---
        output_df = final_df[['Név', 'Total Hours', 'Number of Donations', 'Total Donation Amount', 'Total Wage', 'Donations (Count) / Hour', 'ROI']]
        output_df.rename(columns={'Név': 'Worker Name', 'Total Donation Amount': 'Total Donations (HUF)'}, inplace=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='Worker Performance')
        
        processed_data = output.getvalue()
        return processed_data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure the uploaded files are in the correct format and all columns are present.")
        return None

# --- Streamlit App Interface ---
st.set_page_config(page_title="Donation Statistics Generator", layout="centered")

st.title("📊 Donation Statistics Generator")
st.write("Upload the three required Excel files to generate the performance report.")

donations_file = st.file_uploader("📁 1. Upload Donations File", type=["xlsx"])
names_file = st.file_uploader("📁 2. Upload Names File", type=["xlsx"])
workhours_file = st.file_uploader("📁 3. Upload Work Hours File", type=["xlsx"])

if st.button("🚀 Generate Report"):
    if donations_file and names_file and workhours_file:
        with st.spinner("Processing files... please wait."):
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
