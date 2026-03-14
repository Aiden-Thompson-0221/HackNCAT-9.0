# Standard imports
import sys
import os
import pandas as pd
# Removed OpenAI dependency by default; support OpenAI when `OPENAI_API_KEY` is provided
# Attempt to load environment variables from a .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Configure OpenAI client only if an API key is available in the environment.
# Do NOT hardcode API keys in source files or commit them to the repository.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
try:
    if OPENAI_API_KEY:
        import openai
        openai.api_key = OPENAI_API_KEY
        OPENAI_AVAILABLE = True
    else:
        OPENAI_AVAILABLE = False
except Exception:
    OPENAI_AVAILABLE = False

# --- Load dataset ---
def load_dataset(filepath="Large_Industrial_Pump_Maintenance_Dataset.csv", nrows=None):
    df = pd.read_csv(filepath, nrows=nrows)
    df.rename(columns={
        'Pump_ID': 'pump_id',
        'Temperature': 'temperature',
        'Pressure': 'pressure',
        'Flow_Rate': 'flow_rate',
        'Load_Percent': 'load_percent',
        'Vibration': 'vibration',
        'RPM': 'rpm',
        'Operational_Hours': 'operational_hours',
        'Maintenance_Flag': 'maintenance_flag'
    }, inplace=True)
    # assign a timestamp for this run and ensure pump IDs are unique per row
    df['timestamp'] = pd.Timestamp.now().isoformat()
    df['pump_id'] = df.index.to_series().apply(lambda i: f"PUMP-{i+1:06d}")
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
    # Load percent: higher load can increase risk. Expect 0..100
    try:
        load_percent = float(record.get('load_percent', 0))
    except Exception:
        load_percent = 0.0

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

    # Include load_percent as an additional factor (0..1). Adjusted weights
    # Weights chosen to make 'Moderate' around ~50% typically
    w_temp, w_pressure, w_vib, w_flow, w_load = 0.40, 0.23, 0.14, 0.13, 0.10

    l_score = min(max(load_percent / 100.0, 0.0), 1.0)

    risk_frac = (
        (t_score * w_temp)
        + (p_score * w_pressure)
        + (v_score * w_vib)
        + (f_score * w_flow)
        + (l_score * w_load)
    )
    risk_percent = round(risk_frac * 100, 1)
    return risk_percent

# --- Act Step ---
def create_maintenance_order(record, severity_percent, verbose=False):
    # severity_percent is a numeric percentage (e.g. 50.0)
    if severity_percent >= 75:
        action = "Dispatch maintenance team"
        label = "Critical"
    elif severity_percent >= 40:
        action = "Monitor closely"
        label = "Moderate"
    else:
        action = "Check at next scheduled maintenance"
        label = "Minor"

    work_order = {
        "pump_id": record.get('pump_id'),
        "timestamp": record.get('timestamp'),
        "issue": "Overheating detected",
        "temperature": record.get('temperature'),
        "load_percent": record.get('load_percent'),
        "pressure": record.get('pressure'),
        "flow_rate": record.get('flow_rate'),
        "vibration": record.get('vibration'),
        "rpm": record.get('rpm'),
        "operational_hours": record.get('operational_hours'),
        "maintenance_flag": record.get('maintenance_flag'),
        "severity_percent": severity_percent,
        "severity_label": label,
        "action": action
    }
    return work_order

# --- Explain Step ---
def explain_action(work_order, verbose=False):
    # Backward-compatible local explanation; may be overridden by OpenAI when available
    # Format temperature and load gracefully
    try:
        temp_str = f"{float(work_order.get('temperature')):.2f} C"
    except Exception:
        temp_str = str(work_order.get('temperature'))
    try:
        load_val = work_order.get('load_percent')
        load_str = f"{float(load_val):.1f}%"
    except Exception:
        load_str = str(work_order.get('load_percent'))

    explanation = (
        f"Pump {work_order['pump_id']} reported a temperature of {temp_str} and load of {load_str}.\n"
        f"Severity assessed as: {work_order['severity_label']} ({work_order['severity_percent']}%).\n"
        f"Recommended action: {work_order['action']}."
    )
    if verbose:
        print("\nExplanation:")
        print(explanation)
    return explanation


def generate_openai_explanation(work_order, building_name=None):
    """Generate a detailed, building-specific explanation using OpenAI.
    Falls back to the local `explain_action` when OpenAI is not available or an error occurs.
    """
    if not OPENAI_AVAILABLE:
        return explain_action(work_order, verbose=False)


def _local_detailed_explanation(work_order, building_name=None):
    """Deterministic detailed explanation used as a robust fallback when OpenAI output
    is missing or too short. Produces structured sections similar to the requested format.
    """
    bname = building_name or 'Unknown'
    temp = work_order.get('temperature')
    load = work_order.get('load_percent')
    risk = work_order.get('severity_percent')
    label = work_order.get('severity_label')
    action = work_order.get('action')

    try:
        load_str = f"{float(load):.1f}%"
    except Exception:
        load_str = str(load)

    exec_summary = (
        f"Executive summary:\nPump {work_order.get('pump_id')} in {bname} is reporting a temperature of "
        f"{temp:.2f} C and load of {load_str} with an assessed risk of {risk}% ({label}). The immediate recommendation is: {action}.\n\n"
    )

    temp_analysis = (
        "Detailed temperature analysis:\n"
        f"The recorded temperature of {temp:.2f} C exceeds typical safe operating thresholds (<=90 C). "
        "Elevated temperature can indicate excessive friction, inadequate cooling, blocked flow, or failing bearings. "
        "Thermal stress increases wear rates and can accelerate lubricant breakdown, leading to cascading failures.\n\n"
    )

    immediate = (
        "Immediate short-term actions (in order):\n"
        "1) Safely reduce load or shut down the pump following site safety procedures to prevent further heating.\n"
        "2) Check cooling systems and inlet/outlet flow (valves, heat exchangers) to restore proper flow and cooling.\n"
        "3) Inspect visible components for smoke, leaks, or abnormal noise; wear appropriate PPE and isolate power before hands-on checks.\n\n"
    )

    medium = (
        "Medium-term inspections and repairs:\n"
        "- Inspect bearings and coupling for wear; replace lubrication and bearings if temperatures have been repeatedly high.\n"
        "- Verify impeller condition and clear any blockages in suction strainer or piping.\n"
        "- Test motor and drive for electrical issues causing overheating.\n\n"
    )

    risk_interp = (
        "Risk interpretation and urgency:\n"
        f"A {risk}% risk (label: {label}) suggests a {('low' if risk<40 else 'moderate' if risk<75 else 'high')} near-term probability of progressive failure. "
        "For 'Minor' levels, schedule an inspection within normal maintenance windows; for 'Moderate', prioritize within 24-72 hours; for 'Critical', dispatch immediately.\n\n"
    )

    root_causes = (
        "Likely root causes and components to inspect:\n"
        "- Bearing wear or lubrication failure\n"
        "- Reduced flow due to valve or piping obstruction\n"
        "- Motor electrical issues (overcurrent, poor ventilation)\n\n"
    )

    building_notes = (
        "Building-specific notes and site safety:\n"
        f"For {bname}, confirm access points, isolation valves, and any permit requirements before dispatching staff. Coordinate with site facilities for HVAC or electrical isolation as needed.\n\n"
    )

    personnel = (
        "Recommended personnel, tools, and spare parts:\n"
        "- 1-2 trained pump technicians, one electrical technician if motor issues suspected\n"
        "- Thermal camera, vibration analyzer, basic mechanical toolkit, replacement bearings, lubricants\n\n"
    )

    checklist = (
        "Immediate checklist:\n"
        "[ ] Isolate power and verify lockout/tagout\n"
        "[ ] Reduce load or shutdown if safe\n"
        "[ ] Verify cooling/flow and clear blockages\n"
        "[ ] Record temperature and vibration trend data for follow-up\n\n"
    )

    return exec_summary + temp_analysis + immediate + medium + risk_interp + root_causes + building_notes + personnel + checklist

    # Stronger, structured prompt: system + user messages asking for a long,
    # sectioned, and technical explanation with concrete steps and building-specific notes.
    system_msg = (
        "You are a senior maintenance engineer and technical writer. Produce a very detailed, "
        "technically accurate explanation for the pump report provided. Organize the response with "
        "clear headings, numbered lists, and concise action items. Emphasize safety, concrete "
        "inspection steps, likely root causes, time-to-failure considerations, and recommended "
        "personnel and tools. Mention the building name wherever relevant."
    )

    user_msg = (
        "Create an exhaustive explanation for the pump report below. Include the following sections:\n"
        "- Executive summary (2-3 sentences)\n"
        "- Detailed temperature analysis (why the reading is too hot; what it implies physically)\n"
        "- Immediate short-term actions (3 concrete safety-first steps, in order)\n"
        "- Medium-term inspections and repairs to prevent recurrence\n"
        "- Interpretation of the risk percentage and recommended response time\n"
        "- Likely root causes and components to inspect/replacement suggestions\n"
        "- Building-specific considerations and site safety notes (use building name)\n"
        "- Recommended personnel, tools, and spare parts to dispatch\n"
        "- A final concise checklist summarizing immediate next steps\n\n"
    )

    context = (
        f"Pump ID: {work_order.get('pump_id')}\n"
        f"Building: {building_name or 'Unknown'}\n"
        f"Temperature: {work_order.get('temperature'):.2f} C\n"
        f"Risk: {work_order.get('severity_percent')}% ({work_order.get('severity_label')})\n"
        f"Recommended action (heuristic): {work_order.get('action')}\n"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg + "\n" + context},
    ]

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.1,
            max_tokens=1200,
        )
        text = ""
        if resp and 'choices' in resp and len(resp['choices']) > 0:
            # ChatCompletion returns a message object with 'content'
            text = resp['choices'][0]['message']['content'].strip()
        # If the model returned very little, try a second call with a higher temperature
        if not text or len(text.split()) < 120 or 'Executive summary' not in text:
            try:
                resp2 = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0.35,
                    max_tokens=1500,
                )
                if resp2 and 'choices' in resp2 and len(resp2['choices']) > 0:
                    text2 = resp2['choices'][0]['message']['content'].strip()
                    if text2 and len(text2.split()) >= 120:
                        return text2
            except Exception:
                pass

        if not text or len(text.split()) < 120:
            return _local_detailed_explanation(work_order, building_name=building_name)
        return text
    except Exception:
        return _local_detailed_explanation(work_order, building_name=building_name)

# --- Main Workflow ---
def main():
    sensor_data = load_dataset()
    print("Loaded data rows:", len(sensor_data))
    overheating_pumps = detect_overheating(sensor_data)
    if overheating_pumps.empty:
        print("\nNo overheating detected.")
        return
    
    work_orders = []
    for _, record in overheating_pumps.iterrows():
        severity = ai_assess_severity(record)
        work_order = create_maintenance_order(record, severity, verbose=False)
        # do not print per-record output here; collect for reporting
        work_orders.append(work_order)

    # Allow selection via command-line arg for testing, otherwise prompt the user
    selection = None
    if len(sys.argv) > 1:
        selection = str(sys.argv[1]).strip()
    else:
        print("\nSelect building to view report:")
        print("1) Minor  2) Moderate  3) Critical")
        selection = input("Enter 1, 2, or 3: ").strip()

    severity_map = {'1': 'Minor', '2': 'Moderate', '3': 'Critical'}
    building_map = {'1': 'McNair Hall', '2': 'Gibbs Hall', '3': 'Corbett Gym'}
    if selection not in severity_map:
        print("Invalid selection. Exiting.")
        return

    selected_label = severity_map[selection]
    selected_building = building_map[selection]
    print(f"\nGenerating report for building: {selected_building} (severity: {selected_label})\n")
    print_report(work_orders, selected_label, selected_building)


def print_report(work_orders, severity_label, building_name=None):
    """Print a simple report for the given severity label."""
    if not work_orders:
        print("No work orders available to report.")
        return
    df = pd.DataFrame(work_orders)
    df_filtered = df[df['severity_label'] == severity_label]
    if df_filtered.empty:
        print(f"No incidents with severity '{severity_label}'.")
        return
    # Return only a single pump: pick the highest-risk incident matching label
    top_row = df_filtered.sort_values(['severity_percent', 'temperature'], ascending=[False, False]).iloc[0]
    # Convert to plain dict for downstream functions
    work_order = top_row.to_dict()

    # Print only the requested variables for the selected pump before the report
    print("\nPump variables and values:")
    for key in ['pump_id', 'timestamp', 'temperature', 'pressure', 'flow_rate', 'load_percent']:
        print(f"- {key}: {work_order.get(key)}")

    print("\nSelected pump report:")
    if building_name:
        print(f"Building: {building_name}")
    print(f"Pump ID: {work_order.get('pump_id')}")
    print(f"Timestamp: {work_order.get('timestamp')}")
    try:
        print(f"Temperature: {work_order.get('temperature'):.2f} C")
    except Exception:
        print(f"Temperature: {work_order.get('temperature')}")
    # Print load percent if available
    try:
        lp = work_order.get('load_percent')
        print(f"Load: {float(lp):.1f}%")
    except Exception:
        if work_order.get('load_percent') is not None:
            print(f"Load: {work_order.get('load_percent')}")
    print(f"Risk: {work_order.get('severity_percent')}% ({work_order.get('severity_label')})")
    print(f"Recommended action: {work_order.get('action')}")

    # Generate and print an AI explanation (OpenAI when available, otherwise local fallback)
    explanation = generate_openai_explanation(work_order, building_name=building_name)
    print("\nAI Explanation:")
    print(explanation)
    # If the AI response appears short or insufficient, print a deterministic detailed fallback
    try:
        if isinstance(explanation, str) and len(explanation.split()) < 120:
            print("\nDetailed fallback explanation (local):")
            print(_local_detailed_explanation(work_order, building_name=building_name))
    except Exception:
        pass

if __name__ == "__main__":
    main()