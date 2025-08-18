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

        # --- Sheet 2: Worker Calendar Logic (NEW) ---
        
        # 1. Get a clean list of all shifts
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
        
        # 2. Get a clean list of donation events
        donations_df['Transaction Date'] = pd.to_datetime(donations_df['Transaction Date'], dayfirst=True)
        donations_with_names = pd.merge(donations_df, names_df, left_on='Campaign: Campaign Name', right_on='Campaign name')
        all_donations_df = donations_with_names[['Név', 'Transaction Date']].drop_duplicates()
        all_donations_df.rename(columns={'Transaction Date': 'Date'}, inplace=True)
        
        # 3. Create "event" tables for pivoting
        worked_events = all_shifts_df.copy()
        worked_events['Metric'] = 'Worked'
        worked_events['Value'] = '✓'
        worked_events['Day'] = worked_events['Date'].dt.day

        donated_events = all_donations_df.copy()
        donated_events['Metric'] = 'Donated'
        donated_events['Value'] = '✓'
        donated_events['Day'] = donated_events['Date'].dt.day
        
        # 4. Combine events and pivot to create the calendar
        calendar_events = pd.concat([worked_events, donated_events])
        worker_calendar = calendar_events.pivot_table(
            index=['N
