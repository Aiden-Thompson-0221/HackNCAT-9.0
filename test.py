import pandas as pd
import random
from datetime import datetime
import openai  # Ensure you have openai installed and API key set

# --- Simulate sensor data ---
def generate_sensor_data(num_records=10):
    data = []
    for i in range(num_records):
        record = {
            "timestamp": datetime.now().isoformat(),
            "pump_id": f"PUMP-{random.randint(1,5)}",
            "temperature": random.uniform(50, 120),  # degrees Celsius
            "pressure": random.uniform(1, 5),        # bar
            "flow_rate": random.uniform(10, 100),    # liters/min
            "load_percent": random.uniform(10, 100)  # %
        }
        data.append(record)
    return pd.DataFrame(data)

# --- Detect Step ---
def detect_overheating(df, temp_threshold=90):
    df['overheating'] = df['temperature'] > temp_threshold
    return df[df['overheating']]

# --- AI-Assisted Decide Step ---
def ai_assess_severity(record):
    prompt = f"""
    A pump has the following sensor readings:
    Temperature: {record['temperature']:.2f} C
    Pressure: {record['pressure']:.2f} bar
    Flow Rate: {record['flow_rate']:.2f} L/min
    Load: {record['load_percent']:.2f} %
    
    Is this a minor warning, moderate risk, or critical failure? Respond with one word only.
    """
    # Example using OpenAI's GPT (replace with your own key)
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    severity = response['choices'][0]['message']['content'].strip()
    return severity

# --- Act Step ---
def create_maintenance_order(record, severity):
    work_order = {
        "pump_id": record['pump_id'],
        "timestamp": record['timestamp'],
        "issue": "Overheating detected",
        "temperature": record['temperature'],
        "severity": severity,
        "action": "Dispatch maintenance team" if severity.lower() == "critical" else "Monitor closely"
    }
    print("Maintenance Work Order Created:")
    print(work_order)
    return work_order

# --- Explain Step ---
def explain_action(work_order):
    explanation = (
        f"Pump {work_order['pump_id']} reported a temperature of {work_order['temperature']:.2f} C.\n"
        f"Severity assessed as: {work_order['severity']}.\n"
        f"Recommended action: {work_order['action']}."
    )
    print("\nExplanation:")
    print(explanation)
    return explanation

# --- Main Workflow ---
def main():
    sensor_data = generate_sensor_data(5)
    print("Sensor Data:")
    print(sensor_data)
    
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