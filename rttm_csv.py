import csv
import argparse
import os

def format_minutes(seconds):
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}:{secs:06.3f}"  # format MM:SS.mmm

def rttm_to_csv_with_minutes(rttm_file, output_csv):
    with open(rttm_file, 'r') as fin, open(output_csv, 'w', newline='') as fout:
        reader = csv.reader(fin, delimiter=' ')
        writer = csv.writer(fout)
        writer.writerow(['Speaker', 'Start (min)', 'End (min)', 'Duration (s)', 'Start (s)'])

        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            speaker = row[7]
            start = float(row[3])
            duration = float(row[4])
            end = start + duration

            writer.writerow([
                speaker,
                format_minutes(start),
                format_minutes(end),
                f"{duration:.3f}",
                f"{start:.3f}"
            ])

    print(f"✔ CSV created: {output_csv}")

# Exemple d'appel :
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert RTTM to CSV with minutes format.')
    parser.add_argument('--rttm_file', type=str, help='Path to the input RTTM file', required=True)
    args = parser.parse_args()
    os.makedirs('output_csv', exist_ok=True)  # Create output directory if it doesn't exist

    output_csv = args.rttm_file.split('/')[-1].replace('.rttm', '.csv')
    output_csv = f"output_csv/{output_csv}"
    rttm_to_csv_with_minutes(args.rttm_file, output_csv=output_csv)