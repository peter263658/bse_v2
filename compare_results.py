#!/usr/bin/env python3
"""
Compare evaluation results with values reported in the paper:
"Binaural Speech Enhancement using Deep Complex Convolutional Transformer Networks"

This script generates a comparison table showing your results vs. the paper's values.
"""

import argparse
import pandas as pd
import numpy as np
import os


def compare_with_paper(vctk_results_file, timit_results_file=None, output_file=None):
    """
    Compare evaluation results with values reported in the paper
    
    Args:
        vctk_results_file: Path to VCTK results file
        timit_results_file: Path to TIMIT results file (optional)
        output_file: Path to output file for comparison table
    """
    # Load your results
    vctk_df = pd.read_csv(vctk_results_file)
    
    # Load TIMIT results if provided
    if timit_results_file and os.path.exists(timit_results_file):
        timit_df = pd.read_csv(timit_results_file)
    else:
        timit_df = None
    
    # Paper results for BCCTN model from Table 1 (anechoic condition)
    paper_vctk_results = {
        # Values from Table 1 for BCCTN-Proposed Loss
        -6: {"MBSTOI": 0.73, "SegSNR": 14.3, "ILD": 0.61, "IPD": 8},
        -3: {"MBSTOI": 0.79, "SegSNR": 12.7, "ILD": 0.62, "IPD": 7},
        0: {"MBSTOI": 0.85, "SegSNR": 12.7, "ILD": 0.40, "IPD": 5},
        3: {"MBSTOI": 0.87, "SegSNR": 11.5, "ILD": 0.36, "IPD": 4},
        6: {"MBSTOI": 0.91, "SegSNR": 9.7, "ILD": 0.34, "IPD": 3},
        9: {"MBSTOI": 0.94, "SegSNR": 8.4, "ILD": 0.20, "IPD": 2},
        12: {"MBSTOI": 0.96, "SegSNR": 7.0, "ILD": 0.19, "IPD": 2},
        15: {"MBSTOI": 0.96, "SegSNR": 5.4, "ILD": 0.19, "IPD": 2}
    }
    
    # Paper results for reverberant condition (TIMIT) from Table 2
    paper_timit_results = {
        # Values from Table 2 for BCCTN-Proposed Loss
        -6: {"MBSTOI": 0.66, "SegSNR": 10.3, "ILD": 1.12, "IPD": 12},
        -3: {"MBSTOI": 0.74, "SegSNR": 9.0, "ILD": 0.72, "IPD": 10},
        0: {"MBSTOI": 0.80, "SegSNR": 8.4, "ILD": 0.62, "IPD": 8},
        3: {"MBSTOI": 0.83, "SegSNR": 7.1, "ILD": 0.45, "IPD": 5},
        6: {"MBSTOI": 0.89, "SegSNR": 6.1, "ILD": 0.38, "IPD": 5},
        9: {"MBSTOI": 0.93, "SegSNR": 5.0, "ILD": 0.29, "IPD": 3},
        12: {"MBSTOI": 0.96, "SegSNR": 4.6, "ILD": 0.21, "IPD": 3},
        15: {"MBSTOI": 0.96, "SegSNR": 3.3, "ILD": 0.20, "IPD": 2}
    }
    
    # Create comparison table for VCTK
    comparison_rows = []
    
    # Compare VCTK results
    for snr in sorted(paper_vctk_results.keys()):
        row = vctk_df[vctk_df['SNR'] == snr].iloc[0] if snr in vctk_df['SNR'].values else None
        paper = paper_vctk_results[snr]
        
        if row is not None:
            # MBSTOI
            our_mbstoi = row['MBSTOI']
            paper_mbstoi = paper["MBSTOI"]
            comparison_rows.append({
                'Dataset': 'VCTK',
                'SNR': snr,
                'Metric': 'MBSTOI',
                'Paper Value': paper_mbstoi,
                'Our Value': our_mbstoi,
                'Difference': our_mbstoi - paper_mbstoi
            })
            
            # SegSNR
            our_segsnr = row['ΔSegSNR']
            paper_segsnr = paper["SegSNR"]
            comparison_rows.append({
                'Dataset': 'VCTK',
                'SNR': snr,
                'Metric': 'SegSNR',
                'Paper Value': paper_segsnr,
                'Our Value': our_segsnr,
                'Difference': our_segsnr - paper_segsnr
            })
            
            # ILD Error
            our_ild = row['LILD']
            paper_ild = paper["ILD"]
            comparison_rows.append({
                'Dataset': 'VCTK',
                'SNR': snr,
                'Metric': 'ILD',
                'Paper Value': paper_ild,
                'Our Value': our_ild,
                'Difference': our_ild - paper_ild
            })
            
            # IPD Error
            our_ipd = row['LIPD'] * 180 / np.pi if 'LIPD' in row else row['LIPD']
            paper_ipd = paper["IPD"]
            comparison_rows.append({
                'Dataset': 'VCTK',
                'SNR': snr,
                'Metric': 'IPD',
                'Paper Value': paper_ipd,
                'Our Value': our_ipd,
                'Difference': our_ipd - paper_ipd
            })
    
    # Compare TIMIT results if available
    if timit_df is not None:
        for snr in sorted(paper_timit_results.keys()):
            row = timit_df[timit_df['SNR'] == snr].iloc[0] if snr in timit_df['SNR'].values else None
            paper = paper_timit_results[snr]
            
            if row is not None:
                # MBSTOI
                our_mbstoi = row['MBSTOI']
                paper_mbstoi = paper["MBSTOI"]
                comparison_rows.append({
                    'Dataset': 'TIMIT',
                    'SNR': snr,
                    'Metric': 'MBSTOI',
                    'Paper Value': paper_mbstoi,
                    'Our Value': our_mbstoi,
                    'Difference': our_mbstoi - paper_mbstoi
                })
                
                # SegSNR
                our_segsnr = row['ΔSegSNR']
                paper_segsnr = paper["SegSNR"]
                comparison_rows.append({
                    'Dataset': 'TIMIT',
                    'SNR': snr,
                    'Metric': 'SegSNR',
                    'Paper Value': paper_segsnr,
                    'Our Value': our_segsnr,
                    'Difference': our_segsnr - paper_segsnr
                })
                
                # ILD Error
                our_ild = row['LILD']
                paper_ild = paper["ILD"]
                comparison_rows.append({
                    'Dataset': 'TIMIT',
                    'SNR': snr,
                    'Metric': 'ILD',
                    'Paper Value': paper_ild,
                    'Our Value': our_ild,
                    'Difference': our_ild - paper_ild
                })
                
                # IPD Error
                our_ipd = row['LIPD'] * 180 / np.pi if 'LIPD' in row else row['LIPD']
                paper_ipd = paper["IPD"]
                comparison_rows.append({
                    'Dataset': 'TIMIT',
                    'SNR': snr,
                    'Metric': 'IPD',
                    'Paper Value': paper_ipd,
                    'Our Value': our_ipd,
                    'Difference': our_ipd - paper_ipd
                })
    
    # Create DataFrame from comparison rows
    comparison_df = pd.DataFrame(comparison_rows)
    
    # Calculate averages by metric
    avg_comparison = comparison_df.groupby(['Dataset', 'Metric']).agg({
        'Paper Value': 'mean',
        'Our Value': 'mean',
        'Difference': 'mean'
    }).reset_index()
    
    # Print comparison table
    print("\nComparison with Paper Values:")
    print(comparison_df.to_string(index=False))
    
    print("\nAverage Comparison by Metric:")
    print(avg_comparison.to_string(index=False))
    
    # Save to file if requested
    if output_file:
        comparison_df.to_csv(output_file, index=False)
        avg_output_file = os.path.splitext(output_file)[0] + "_averages.csv"
        avg_comparison.to_csv(avg_output_file, index=False)
        print(f"\nComparison saved to {output_file}")
        print(f"Average comparison saved to {avg_output_file}")
    
    return comparison_df, avg_comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare evaluation results with paper values")
    parser.add_argument("--vctk_results", required=True, help="Path to VCTK results CSV file")
    parser.add_argument("--timit_results", help="Path to TIMIT results CSV file (optional)")
    parser.add_argument("--output_file", help="Path to output file for comparison table")
    
    args = parser.parse_args()
    
    compare_with_paper(args.vctk_results, args.timit_results, args.output_file)
