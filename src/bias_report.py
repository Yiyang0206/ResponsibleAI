import pandas as pd

def identify_bias_in_fairness_metrics(fairness_df, dpd_threshold=0.2, eod_threshold=0.2, di_threshold=0.8):
    """Identify potential fairness issues based on DPD, EOD, and DI metrics."""
    
    # Lists to capture flagged groups for each type of bias
    dpd_concerns = []
    eod_concerns = []
    di_concerns = []
    
    # Check each group for potential fairness issues
    for _, row in fairness_df.iterrows():
        group = row['Group_code']
        group_size = row['Group Size']
        dpd = row['Demographic Parity Difference']
        eod = row['Equal Opportunity Difference']
        di = row['Disparate Impact']
        
        # Check for potential issues with Demographic Parity Difference
        if abs(dpd) > dpd_threshold:
            dpd_concerns.append({
                'Group': group,
                'Group Size': group_size,
                'DPD': dpd
            })
        
        # Check for potential issues with Equal Opportunity Difference
        if abs(eod) > eod_threshold:
            eod_concerns.append({
                'Group': group,
                'Group Size': group_size,
                'EOD': eod
            })
        
        # Check for potential issues with Disparate Impact
        if di < di_threshold:
            di_concerns.append({
                'Group': group,
                'Group Size': group_size,
                'DI': di
            })
    
    # Combine all flagged concerns into a single summary
    bias_report = {
        'Demographic Parity Concerns': dpd_concerns,
        'Equal Opportunity Concerns': eod_concerns,
        'Disparate Impact Concerns': di_concerns
    }
    
    # Output the bias report
    return bias_report

def print_bias_report(bias_report):
    """Prints a summary of the bias report."""
    
    print("Bias Concerns Report\n" + "="*50)
    
    # Print DPD concerns
    print("\nDemographic Parity Concerns (DPD > 0.2):")
    if bias_report['Demographic Parity Concerns']:
        for concern in bias_report['Demographic Parity Concerns']:
            print(f"Group: {concern['Group']} (Size: {concern['Group Size']}), DPD: {concern['DPD']}")
    else:
        print("No concerns identified.")
    
    # Print EOD concerns
    print("\nEqual Opportunity Concerns (EOD > 0.2):")
    if bias_report['Equal Opportunity Concerns']:
        for concern in bias_report['Equal Opportunity Concerns']:
            print(f"Group: {concern['Group']} (Size: {concern['Group Size']}), EOD: {concern['EOD']}")
    else:
        print("No concerns identified.")
    
    # Print DI concerns
    print("\nDisparate Impact Concerns (DI < 0.8):")
    if bias_report['Disparate Impact Concerns']:
        for concern in bias_report['Disparate Impact Concerns']:
            print(f"Group: {concern['Group']} (Size: {concern['Group Size']}), DI: {concern['DI']}")
    else:
        print("No concerns identified.")

def report(dataset_name):
    fairness_df = pd.read_csv(f'results/{dataset_name}/fairness_analysis.csv')
    bias_report = identify_bias_in_fairness_metrics(fairness_df)
    print_bias_report(bias_report)

report('compas')
report('census')
report('credit')


