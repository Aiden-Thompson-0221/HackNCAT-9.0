import pandas as pd
# Removed OpenAI dependency; using local heuristic and dataset instead

# --- Load dataset ---
def load_dataset(filepath="Large_Industrial_Pump_Maintenance_Dataset.csv", nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    df.rename(columns={
        'Pump_ID': 'pump_id',
        'Temperature': 'temperature',
        'Pressure': 'pressure',
        'Flow_Rate': 'flow_rate',
        'Vibration': 'vibration',
        'RPM': 'rpm',
        'Operational_Hours': 'operational_hours',
        'Maintenance_Flag': 'maintenance_flag'
    }, inplace=True)
    df['timestamp'] = pd.Timestamp.now().isoformat()
    return df

# --- Detect Step ---
def detect_overheating(df, temp_threshold=90):
    df['overheating'] = df['temperature'] > temp_threshold
    return df[df['overheating']]

# --- AI-Assisted Decide Step ---
def ai_assess_severity(record):
    # Return a percentage risk score (0-100) based on normalized signals.
    temp = float(record.get('temperature', 0))
    pressure = float(record.get('pressure', 0))
    vibration = float(record.get('vibration', 0)) if 'vibration' in record else 0
    flow = float(record.get('flow_rate', 0))

    # Normalize each metric to 0..1 where 1 is highest risk.
    # Temperature: <=90 -> 0, 90-130 -> linear 0..1, >130 -> 1
    if temp <= 90:
        t_score = 0.0
    elif temp >= 130:
        t_score = 1.0
    else:
        t_score = (temp - 90) / (130 - 90)

    # Pressure: assume operational range 100..300; higher pressure -> higher risk
    p_min, p_max = 100.0, 300.0
    p_score = min(max((pressure - p_min) / (p_max - p_min), 0.0), 1.0)

    # Vibration: >=5 -> 1, else scale 0..1 (0..5)
    v_score = min(max(vibration / 5.0, 0.0), 1.0)

    # Flow: low flow is risky. Map flow 0..20 -> 1..0 (inverse)
    f_score = 1.0 - min(max(flow / 20.0, 0.0), 1.0)

    # Weights chosen to make 'Moderate' around ~50% typically
    w_temp, w_pressure, w_vib, w_flow = 0.45, 0.25, 0.15, 0.15

    risk_frac = (t_score * w_temp) + (p_score * w_pressure) + (v_score * w_vib) + (f_score * w_flow)
    risk_percent = round(risk_frac * 100, 1)
    return risk_percent

# --- Act Step ---
def create_maintenance_order(record, severity_percent):
    # severity_percent is a numeric percentage (e.g. 50.0)
    if severity_percent >= 75:
        action = "Dispatch maintenance team"
        label = "Critical"
    elif severity_percent >= 40:
        action = "Monitor closely"
        label = "Moderate"
    else:
        action = "Monitor closely"
        label = "Minor"

    work_order = {
        "pump_id": record.get('pump_id'),
        "timestamp": record.get('timestamp'),
        "issue": "Overheating detected",
        "temperature": record.get('temperature'),
        "severity_percent": severity_percent,
        "severity_label": label,
        "action": action
    }
    print("Maintenance Work Order Created:")
    print(work_order)
    return work_order

# --- Explain Step ---
def explain_action(work_order):
    explanation = (
        f"Pump {work_order['pump_id']} reported a temperature of {work_order['temperature']:.2f} C.\n"
        f"Severity assessed as: {work_order['severity_label']} ({work_order['severity_percent']}%).\n"
        f"Recommended action: {work_order['action']}."
    )
    print("\nExplanation:")
    print(explanation)
    return explanation

# --- Main Workflow ---
def main():
    sensor_data = load_dataset()
    print("Loaded data rows:", len(sensor_data))
    overheating_pumps = detect_overheating(sensor_data)
    if overheating_pumps.empty:
        print("\nNo overheating detected.")
        return
    
    for _, record in overheating_pumps.iterrows():
        severity = ai_assess_severity(record)
        work_order = create_maintenance_order(record, severity)
        explain_action(work_order)

if __name__ == "__main__":
    main()