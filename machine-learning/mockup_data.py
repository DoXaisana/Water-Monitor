import random
import datetime
import pandas as pd
import os
from calendar import monthrange

class LaosWaterUsageSimulator:
    def __init__(self, household_size=4, output_dir="datasets"):
        self.household_size = household_size
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.size_multiplier = {
            1: 0.4, 2: 0.7, 3: 0.9, 4: 1.0, 5: 1.2, 6: 1.4
        }[household_size]
        
        self.seasonal_multipliers = {
            1: 0.8, 2: 0.9, 3: 1.1, 4: 1.2, 5: 1.1, 6: 1.0,
            7: 1.0, 8: 1.0, 9: 1.0, 10: 0.9, 11: 0.8, 12: 0.8
        }

    def is_special_event(self, date):
        events = []
        if date.month == 4 and date.day in [13, 14, 15]:
            events.append(("Lao New Year", 1.6))
        if random.random() < 0.05:
            event_types = [
                ("House Party", 1.5), ("Family Gathering", 1.3),
                ("House Cleaning", 1.4), ("Visitors", 1.3)
            ]
            events.append(random.choice(event_types))
        if random.random() < 0.02:
            events.append(("Vacation", 0.2))
        return events

    def get_usage_percentage(self, hour, is_weekend, events):
        base_usage = {
            "weekday": {
                "morning_peak": (5, 9, 30, 45), "midday": (9, 12, 10, 20),
                "lunch": (12, 14, 30, 50), "afternoon": (14, 18, 10, 25),
                "evening_peak": (18, 22, 50, 70), "night": (22, 5, 5, 15)
            },
            "weekend": {
                "morning_peak": (7, 11, 50, 70), "midday": (11, 13, 30, 50),
                "lunch": (13, 15, 40, 60), "afternoon": (15, 18, 30, 50),
                "evening_peak": (18, 23, 60, 80), "night": (23, 7, 5, 15)
            }
        }

        day_type = "weekend" if is_weekend else "weekday"
        patterns = base_usage[day_type]
        base = 10

        for period, (start, end, min_usage, max_usage) in patterns.items():
            if (start <= hour < end) or (start > end and (hour >= start or hour < end)):
                base = random.uniform(min_usage, max_usage)
                break

        event_multiplier = 1.0
        for event, multiplier in events:
            event_multiplier *= multiplier

        return base * event_multiplier

    def generate_monthly_data(self, year, month):
        start_date = datetime.date(year - 2, month, 1)
        days_in_month = monthrange(year - 2, month)[1]
        end_date = start_date + datetime.timedelta(days=days_in_month)

        data = []
        for day in range(days_in_month):
            current_date = start_date + datetime.timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            events = self.is_special_event(current_date)
            seasonal_multiplier = self.seasonal_multipliers[current_date.month]
            
            for interval in range(48):
                time = (datetime.datetime.combine(current_date, datetime.time.min) + 
                       datetime.timedelta(minutes=interval * 30)).time()
                
                base_usage = self.get_usage_percentage(time.hour, is_weekend, events)
                actual_usage = base_usage * seasonal_multiplier * self.size_multiplier
                actual_usage *= random.uniform(0.9, 1.1)
                actual_usage = round(max(0, min(actual_usage, 100)), 2)

                if actual_usage == 100:
                    actual_usage -= random.randint(8, 15)

                convert_to_liters = round((actual_usage * 1000) / 100, 2)
                event_names = "; ".join([e[0] for e in events]) if events else "Normal day"

                data.append([
                    current_date, time, actual_usage, convert_to_liters, event_names, is_weekend
                ])
        
        return pd.DataFrame(data, columns=[
            'Date', 'Time', 'Usage Percentage', 'Convert Liters', 'Special Events', 'Is Weekend'
        ])

    def generate_yearly_data(self, year):
        for month in range(1, 13):  # Fixed loop range to include all months
            monthly_data = self.generate_monthly_data(year, month)
            filename = f"water_usage_{year - 2}_{month:02d}.csv"
            filepath = os.path.join(self.output_dir, filename)
            monthly_data.to_csv(filepath, index=False)
            print(f"Generated data for {year - 2}-{month:02d} saved to {filename}")

# Example usage
if __name__ == "__main__":
    simulator = LaosWaterUsageSimulator(household_size=3)
    current_year = datetime.date.today().year
    simulator.generate_yearly_data(current_year)