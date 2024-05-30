import simpy
import random
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import numpy as np
import faker as Faker
import uuid
from faker import Faker
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go


# Constants
NUM_GENERAL_ROOMS = 1
NUM_SURGERY_ROOMS = 1

patient_data = []
patients_appt = []
staff_id_list = []
summary_data = []


patient_id = 0
patients_in_queue = 0
num_patients_treated = 0
room_availability = "None"


available_rooms = 0
general_room_occupancy = 0
surgery_room_occupancy = 0


num_in_general_rooms = 0
num_in_surgery_rooms = 0
general_room_next_free = 0
surgery_room_next_free = 0

# Track statistics
total_wait_time_general = 0
total_wait_time_surgery = 0
total_patients_general = 0
total_patients_surgery = 0

# Initialize Faker
fake = Faker()

def generate_staff_identity_data(num_doctors, num_nurses_general, num_nurses_surgery, doctor_shifts, nurse_shifts):
    staff_identity_data = []
    # Function to generate random age between 25 and 65
    def generate_age():
        return np.random.randint(25, 66)

    # Function to generate random years of experience between 1 and 30
    def generate_experience():
        return np.random.randint(1, 31)

    # Function to generate random cost per hour between 50 and 200
    def generate_cost_per_hour():
        return round(np.random.uniform(50, 200), 2)

    # Generate data for doctors
    for _ in range(num_doctors):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = doctor_shifts
        role = "Doctor"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])

    # Generate data for nurses
    for _ in range(num_nurses_general):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = nurse_shifts
        role = "Nurse"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])
        
    for _ in range(num_nurses_surgery):
        staff_id = str(uuid.uuid4())
        staff_name = fake.name()
        age = generate_age()
        years_of_experience = generate_experience()
        cost_per_hour = generate_cost_per_hour()
        shift = nurse_shifts
        role = "Nurse"
        staff_identity_data.append([staff_id, staff_name, role, age, years_of_experience, cost_per_hour, shift])

    # Create DataFrame
    staff_identity_df = pd.DataFrame(staff_identity_data, columns=["staff_id", "staff_name", "role", "age", "years_of_experience", "cost_per_hour", "shifts"])
    
    return staff_identity_df


def simulate_staff_schedule(num_days, start_datetime):
    # Initialize schedule dictionary
    schedule = []
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    # Generate staff identity data
    staff_identity_df = generate_staff_identity_data(num_doctors, num_nurses_general, num_nurses_surgery, doctor_shifts, nurse_shifts)
    # Generate schedule for each weekday
    for days in range(num_days):
        current_date = start_datetime + timedelta(days=days)

        if current_date.strftime("%A") in weekdays:
            for procedure_type in ["General Checkup", "Surgery"]:    
                # Assign doctors and nurses randomly to each procedure
                doctors = staff_identity_df[staff_identity_df['role'] == 'Doctor']
                nurses = staff_identity_df[staff_identity_df['role'] == 'Nurse']

                assigned_doctor = doctors.sample(1).iloc[0]
                num_nurses = num_nurses_surgery if procedure_type == "Surgery" else num_nurses_general
                assigned_nurses = nurses.sample(num_nurses)

                schedule.append({
                    "scheduled_date": current_date.strftime('%Y-%m-%d'),
                    "day": current_date.strftime("%A"),
                    "procedure_type": procedure_type,
                    "doctor_id": assigned_doctor['staff_id'],
                    "doctor_name": f"Dr. {assigned_doctor['staff_name']}",
                    "nurse_name(s)": [nurse['staff_name'] for _, nurse in assigned_nurses.iterrows()],
                    "doctor_shift_hours": assigned_doctor['shifts'],
                    "total_nurse_shift_hours": assigned_nurses['shifts'].sum()
                })

    staff_month_schedule = pd.DataFrame(schedule)
    staff_month_schedule.to_csv('staff_month_schedules.csv', index=False)
    return staff_month_schedule

def get_patient_appointment(env, start_datetime, arrival_time_min):
    
    global general_room_next_free, surgery_room_next_free, patient_data, patients_in_queue, patient_id
    
    # Calculate the arrival datetime
    arrival_datetime = start_datetime + timedelta(minutes=arrival_time_min)
    
    # Randomize delay between arrival and treatment start time (30 minutes to 1 hour)
    delay = random.randint(30, 60)
    
    start_time_minutes = (start_datetime - start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 60
    
    # Convert arrival_datetime to minutes
    arrival_time_minutes = (arrival_datetime - start_datetime).total_seconds() / 60
    # Calculate treatment start time in minutes
    treatment_start_time_minutes = max(arrival_time_minutes + delay, (env.now - start_time_minutes))
    # Convert treatment start time back to a datetime object
    treatment_start_time = start_datetime + timedelta(minutes=treatment_start_time_minutes)
        
    # Decide the room type (General or Surgery)
    room_type = random.choice(['General Checkup', 'Surgery'])

    
    if room_type == 'General Checkup':
        # Wait for the general room to be available
        if env.now < general_room_next_free:
            patients_in_queue += 1
            yield env.timeout(max(0, general_room_next_free - env.now))
            if patients_in_queue > 0:
                patients_in_queue -= 1
        
        wait_time = treatment_start_time_minutes - arrival_time_minutes  # Calculate the wait time
        duration = GENERAL_DURATION
        general_room_next_free = env.now + duration  # Update next available time
        yield env.timeout(duration)  # Simulate the duration of the procedure
    else:
        # Wait for the surgery room to be available
        if env.now < surgery_room_next_free:
            patients_in_queue += 1
            yield env.timeout(max(0, surgery_room_next_free - env.now))
            if patients_in_queue > 0:
                patients_in_queue -= 1

        wait_time = treatment_start_time_minutes - arrival_time_minutes  # Calculate the wait time
        duration = SURGERY_DURATION
        surgery_room_next_free = env.now + duration  # Update next available time
        yield env.timeout(duration)  # Simulate the duration of the procedure
    
    treatment_end_time = treatment_start_time + timedelta(minutes=duration)

    patient_id += 1

    patient_data.append({
        'patient_id': patient_id,
        'procedure_type': room_type,
        'arrival_time': arrival_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'treatment_start_time': treatment_start_time,
        'treatment_end_time': treatment_end_time,
        'wait_time': wait_time,
        'duration': duration
    })
    
    return wait_time, duration

def get_patients_df():
    patient_df = pd.DataFrame(patient_data)
    patient_df.to_csv('patients_appointments.csv', index=False)

    return patient_df


def simulate_hospital(sim_duration_days):
    global env, patient_id, total_wait_time_general, total_wait_time_surgery, total_patients_general, total_patients_surgery
    for _ in range(sim_duration_days * 24):
        for _ in range(60 // wait_time_interval):
            num_arrivals = random.randint(1, 3)
            for _ in range(num_arrivals):
                # arrival_time = env.now + random.uniform(0, wait_time_interval)  # spread arrivals within the time interval
                arrival_time_minutes = env.now / 60 + random.uniform(0, wait_time_interval)  # Convert seconds to minutes
                # arrival_time_timedelta = timedelta(minutes=arrival_time_minutes)  # Create a timedelta object
                # arrival_time = start_datetime + arrival_time_timedelta  # Add timedelta to start_datetime
                summary_data = yield env.process(patient_arrivals(env, start_datetime, wait_time_interval, arrival_time_minutes))
                print("Patients Summary Dataframe: ", pd.DataFrame(summary_data));
    
    
# Calculate averages    
def calculate_averages():
    global total_wait_time_general, total_wait_time_surgery, total_patients_general, total_patients_surgery
    avg_wait_time_general = total_wait_time_general / total_patients_general if total_patients_general > 0 else 0
    avg_wait_time_surgery = total_wait_time_surgery / total_patients_surgery if total_patients_surgery > 0 else 0
    return avg_wait_time_general, avg_wait_time_surgery
   

def patient_arrivals(env, start_datetime, wait_time_interval, arrival_time_minutes):
    global patients_in_queue, total_wait_time_general, total_wait_time_surgery, patient_id
    global total_patients_general, total_patients_surgery, summary_data
    global general_room_next_free, surgery_room_next_free, general_room_occupancy, surgery_room_occupancy    

    sim_duration_minutes = sim_duration_days * 24 * 60
    current_time = start_datetime + timedelta(minutes=env.now)

    for minute in range(0, sim_duration_minutes, wait_time_interval):
        num_new_patients = random.randint(1, 3)
        yield env.timeout(wait_time_interval * 60)  # wait_time_interval is in minutes, so convert to seconds

        for _ in range(num_new_patients):
            # wait_time, duration = yield env.process(get_patient_appointment(env, current_time, arrival_time_minutes))
            appointment_result = yield env.process(get_patient_appointment(env, current_time, arrival_time_minutes))
            wait_time, duration = appointment_result
            print("Wait time:", wait_time)
            print("Duration:", duration)


            if duration == GENERAL_DURATION:
                total_wait_time_general += wait_time
                total_patients_general += 1
            else:
                total_wait_time_surgery += wait_time
                total_patients_surgery += 1                

            max_available_rooms = NUM_GENERAL_ROOMS + NUM_SURGERY_ROOMS
            available_rooms = 0

            if env.now >= general_room_next_free and available_rooms < max_available_rooms and general_room_occupancy < NUM_GENERAL_ROOMS:
                available_rooms += 1
                if patients_in_queue > 0:
                    patients_in_queue -= 1

            if env.now >= surgery_room_next_free and available_rooms < max_available_rooms and surgery_room_occupancy < NUM_SURGERY_ROOMS:
                available_rooms += 1
                if patients_in_queue > 0:
                    patients_in_queue -= 1

            room_availability = "None"
            if available_rooms > 0:
                if general_room_occupancy < NUM_GENERAL_ROOMS:
                    room_availability = "General Room Available"
                if surgery_room_occupancy < NUM_SURGERY_ROOMS:
                    room_availability = "Surgery Room Available"
                if general_room_occupancy < NUM_GENERAL_ROOMS and surgery_room_occupancy < NUM_SURGERY_ROOMS:
                    room_availability = "Both Available"

            summary_data.append({
                'scheduled_date': current_time.strftime('%Y-%m-%d'),
                'time_slot': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'new_patients_arrival': num_new_patients,
                'patients_in_queue': patients_in_queue,
                'patients_moved_into_room': min(num_new_patients, NUM_GENERAL_ROOMS + NUM_SURGERY_ROOMS - (general_room_occupancy + surgery_room_occupancy)),
                'room_availability': room_availability,
                'available_rooms_count': available_rooms
            })
        
        current_time += timedelta(days=1)
        print("Scheduled date", current_time)
        print("time_slot", current_time.strftime('%Y-%m-%d %H:%M:%S'))
        current_time += timedelta(minutes=wait_time_interval)  # Increment the time slot
    return summary_data

                
def get_summary_df():
    patient_summary_df= pd.DataFrame(summary_data)
    print(patient_summary_df)
    patient_summary_df.to_csv('patients_schedule.csv', index=False)
    return patient_summary_df     


# Streamlit UI
st.title("Hospital Operations Scheduling Simulation")

#Select number of doctors and nurses
num_doctors = st.sidebar.slider("Number of Doctors", min_value=1, max_value=6, value=3)
num_nurses_general = st.sidebar.slider("Number of Nurses (General)", min_value=1, max_value=4, value=2)
num_nurses_surgery =st.sidebar.slider("Number of Nurses (Surgery)", min_value=2, max_value=8, value=4)

#Set mean duration of procedures general/surgery 
GENERAL_DURATION = st.sidebar.slider("Mean Duration of General Consultation", min_value=30, max_value=60, value=45)
SURGERY_DURATION = st.sidebar.slider("Mean Duration of Surgery Procedure", min_value=60, max_value=120, value=90)

sim_duration_days = st.sidebar.slider("Simulation Duration (Days)", min_value=1, max_value=31, value=7)

#Select time slot min
wait_time_interval = st.sidebar.slider("Time Slot Intervals (in min)", min_value=0, max_value=60, value=15)

# start_date = st.date_input("Start Date", min_value=datetime(2024, 5, 1), max_value=datetime(2024, 5, 31), value=datetime(2024, 5, 2))
doctor_shifts = st.sidebar.slider("Doctor Shift (in hours)", min_value=3, max_value=8, value=5)
nurse_shifts = st.sidebar.slider("Nurse Shift (in hours)", min_value=7, max_value=14, value=8)

# start_datetime = datetime.now()
start_datetime = datetime(2024, 5, 1, 0, 0, 0)

env = simpy.Environment()
env.process(simulate_hospital(sim_duration_days))
env.run(until=sim_duration_days * 24 * 60)

# Simulate button
if st.sidebar.button("Simulate"):
    staff_month_schedule = simulate_staff_schedule(sim_duration_days, start_datetime)
    st.write("Doctors/Nurses Staff Schedule:")
    st.write(staff_month_schedule)
    
    st.write("Patients Summary Schedule")
    patients_summary_df = get_summary_df()
    st.write(patients_summary_df)

    avg_wait_time_general, avg_wait_time_surgery = calculate_averages()

    st.write(f"Average Wait Time for General Checkup: {avg_wait_time_general:.2f} minutes")
    st.write(f"Average Wait Time for Surgery: {avg_wait_time_surgery:.2f} minutes")
    
    # Plot Average Wait Time for General Checkup and Surgery
    wait_times = {'General Checkup': avg_wait_time_general, 'Surgery': avg_wait_time_surgery}
    fig, ax = plt.subplots()
    ax.bar(wait_times.keys(), wait_times.values())
    ax.set_title('Average Wait Time')
    ax.set_ylabel('Minutes')
    st.pyplot(fig)
    
    #Load patient data
    patients_df = get_patients_df()
    st.write("Patients Appointments Simulation")
    st.write(patients_df)

    # Merge patient data with staff schedule based on procedure type
    merge_patients_staff_df = pd.merge(staff_month_schedule, patients_df, on='procedure_type', how='inner')

    # Calculate average wait time per doctor
    avg_wait_time_per_doctor = merge_patients_staff_df.groupby('doctor_name')['wait_time'].mean()
    st.write("Average Wait Time per Doctor:")
    st.write(avg_wait_time_per_doctor)
    
    # Plot Average Wait Time per Doctor
    fig, ax = plt.subplots()
    avg_wait_time_per_doctor.plot(kind='bar', ax=ax)
    ax.set_title('Average Wait Time per Doctor')
    ax.set_ylabel('Minutes')
    st.pyplot(fig)

    


    

    
   



