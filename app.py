import streamlit as st
import pandas as pd
import numpy as np
import io


# Function to process the dataframes and return the final excel file in memory
def process_files(donations_file, names_file, workhours_file):
    try:
        # Load the data from the uploaded files
        # The user fixed skiprows to 14 in the last version.
        donations_df = pd.read_excel(donations_file, skiprows=14)
        names_df = pd.read_excel(names_file)
        workhours_df = pd.read_excel(workhours_file, header=None)

        # --- Clean and Process Donations Data ---
        donations_df.columns = donations_df.columns.str.strip()
        donations_df = donations_df.iloc[:-3]
        donations_df['Amount'] = pd.to_numeric(donations_df['Amount'], errors='coerce')

        donations_sum = donations_df.groupby('Campaign: Campaign Name')['Amount'].sum().reset_index()
        donations_sum.rename(columns={'Campaign: Campaign Name': 'Campaign name', 'Amount': 'Total Donation Amount'},
                             inplace=True)

        donations_count = donations_df.groupby('Campaign: Campaign Name').size().reset_index(name='Number of Donations')
        donations_count.rename(columns={'Campaign: Campaign Name': 'Campaign name'}, inplace=True)

        # --- Calculate Total Hours Worked ---
        name_columns = [4, 5, 6]
        hours_per_person = {}
        for col in name_columns:
            cleaned_series = workhours_df[col].astype(str).str.strip()
            counts = cleaned_series.value_counts()
            for name, count in counts.items():
                if name not in ['---', 'nan']:
                    hours_per_person[name] = hours_per_person.get(name, 0) + count
        total_hours_worked = {name: count * 4 for name, count in hours_per_person.items()}
        hours_df = pd.DataFrame(list(total_hours_worked.items()), columns=['Név', 'Total Hours'])

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
        final_df['ROI'] = np.divide((final_df['Total Donation Amount'] - final_df['Total Wage']),
                                    final_df['Total Wage'])
        final_df.loc[final_df['Total Wage'] == 0, 'ROI'] = np.inf
        final_df.loc[(final_df['Total Wage'] == 0) & (final_df['Total Donation Amount'] == 0), 'ROI'] = 0

        # --- Format and Prepare Output ---
        output_df = final_df[['Név', 'Total Hours', 'Number of Donations', 'Total Donation Amount', 'Total Wage',
                              'Donations (Count) / Hour', 'ROI']]
        output_df.rename(columns={'Név': 'Worker Name', 'Total Donation Amount': 'Total Donations (HUF)'}, inplace=True)

        # NOTE: Formatting is removed here to keep data numeric for Excel. Excel can handle the formatting.

        # Convert dataframe to an in-memory Excel file
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='Worker Performance')

        processed_data = output.getvalue()
        return processed_data

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None


# --- Streamlit App Interface ---
st.set_page_config(page_title="Donation Statistics Generator", layout="centered")

st.title("📊 Donation Statistics Generator")
st.write("Upload the three required Excel files to generate the performance report.")

# Create file uploaders
donations_file = st.file_uploader("📁 1. Upload Donations File", type=["xlsx"])
names_file = st.file_uploader("📁 2. Upload Names File", type=["xlsx"])
workhours_file = st.file_uploader("📁 3. Upload Work Hours File", type=["xlsx"])

# Add a button to run the analysis
if st.button("🚀 Generate Report"):
    if donations_file and names_file and workhours_file:
        st.info("Processing files... please wait.")
        # Process the files
        excel_data = process_files(donations_file, names_file, workhours_file)

        if excel_data:
            st.success("✅ Report generated successfully!")
            # Create a download button
            st.download_button(
                label="📥 Download Report",
                data=excel_data,
                file_name="worker_statistics_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning("⚠️ Please upload all three files to generate the report.")