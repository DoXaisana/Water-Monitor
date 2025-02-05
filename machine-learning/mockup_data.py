import random
import datetime
import pandas as pd
import os

class LaosWaterUsageSimulator:
    def __init__(self, household_size=4, output_dir="datasets"):
        self.household_size = household_size
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Base multiplier for household size
        self.size_multiplier = {
            1: 0.4,    # Single person
            2: 0.7,    # Couple
            3: 0.9,    # Small family
            4: 1.0,    # Average family
            5: 1.2,    # Large family
            6: 1.4     # Extended family
        }[household_size]
        
        # Seasonal temperature variations in Laos
        self.seasonal_multipliers = {
            1: 0.8,    # January (cool)
            2: 0.9,    # February (warming)
            3: 1.1,    # March (hot)
            4: 1.2,    # April (hottest)
            5: 1.1,    # May (hot)
            6: 1.0,    # June (wet)
            7: 1.0,    # July (wet)
            8: 1.0,    # August (wet)
            9: 1.0,    # September (wet)
            10: 0.9,   # October (cooling)
            11: 0.8,   # November (cool)
            12: 0.8    # December (cool)
        }

    def is_special_event(self, date):
        """Check for special events including Lao New Year."""
        events = []
        
        # Lao New Year (Pi Mai Lao)
        if date.month == 4 and date.day in [13, 14, 15]:
            events.append(("Lao New Year", 1.6))  # Double water usage during New Year
        
        # Random events
        if random.random() < 0.05:  # 5% chance of special event
            event_types = [
                ("House Party", 1.5),
                ("Family Gathering", 1.3),
                ("House Cleaning", 1.4),
                ("Visitors", 1.3)
            ]
            events.append(random.choice(event_types))
        
        # Vacation (reduced usage)
        if random.random() < 0.02:  # 2% chance of being on vacation
            events.append(("Vacation", 0.2))
            
        return events

    def get_usage_percentage(self, hour, is_weekend, events):
        """Simulates daily water usage behavior based on time of day and special conditions."""
        base_usage = {
            # Weekday patterns
            "weekday": {
                "morning_peak": (5, 9, 30, 45),    # Earlier start for workdays
                "midday": (9, 12, 10, 20),
                "lunch": (12, 14, 30, 50),
                "afternoon": (14, 18, 10, 25),
                "evening_peak": (18, 22, 50, 70),
                "night": (22, 5, 5, 15)
            },
            # Weekend patterns
            "weekend": {
                "morning_peak": (7, 11, 50, 70),   # Later start, higher usage
                "midday": (11, 13, 30, 50),
                "lunch": (13, 15, 40, 60),
                "afternoon": (15, 18, 30, 50),
                "evening_peak": (18, 23, 60, 80),
                "night": (23, 7, 5, 15)
            }
        }

        # Select day type
        day_type = "weekend" if is_weekend else "weekday"
        patterns = base_usage[day_type]

        # Determine base usage based on time of day
        base = 10  # Default base usage
        for period, (start, end, min_usage, max_usage) in patterns.items():
            if (start <= hour < end) or (start > end and (hour >= start or hour < end)):
                base = random.uniform(min_usage, max_usage)
                break

        # Apply multipliers for events
        event_multiplier = 1.0
        for event, multiplier in events:
            event_multiplier *= multiplier

        return base * event_multiplier

    def generate_monthly_data(self, year, month):
        """Generate data for a specific month."""
        # Create start and end dates for the month
        start_date = datetime.date(year, month, 1)
        if month == 12:
            end_date = datetime.date(year, 1, 1)
        else:
            end_date = datetime.date(year, month + 1, 1)
        
        days = (end_date - start_date).days
        data = []
        
        for day in range(days):
            current_date = start_date + datetime.timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            events = self.is_special_event(current_date)
            
            # Get seasonal multiplier
            seasonal_multiplier = self.seasonal_multipliers[current_date.month]
            
            for interval in range(48):  # 48 intervals per day (every 30 minutes)
                time = (datetime.datetime.combine(current_date, datetime.time.min) + 
                       datetime.timedelta(minutes=interval * 30)).time()
                
                # Get base usage for this time
                base_usage = self.get_usage_percentage(time.hour, is_weekend, events)
                
                # Apply multipliers and add some random variation
                actual_usage = base_usage * seasonal_multiplier * self.size_multiplier
                actual_usage *= random.uniform(0.9, 1.1)  # Add 10% random variation
                
                # Keep within realistic bounds
                actual_usage = round(max(0, min(actual_usage, 100)), 2)

                if actual_usage == 100:
                    actual_usage -= random.randint(8,15)

                convert_to_litter = (actual_usage * 1000) / 100

                # Set decimal 2 point
                convert_to_litter = round(convert_to_litter, 2)
                
                # Record any special events for the day
                event_names = "; ".join([e[0] for e in events]) if events else "Normal day"
                
                data.append([
                    current_date,
                    time,
                    actual_usage,
                    convert_to_litter,
                    event_names,
                    is_weekend
                ])
        
        return pd.DataFrame(data, columns=[
            'Date', 'Time', 'Usage Percentage', 'Convert Litter', 'Special Events', 'Is Weekend'
        ])

    def generate_yearly_data(self, year):
        """Generate and save data for each month of the year."""
        for month in range(1, 5):
            # Generate data for the month
            monthly_data = self.generate_monthly_data(year, month)
            
            # Create filename with year and month
            filename = f"water_usage_{year}_{month:02d}.csv"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save to CSV
            monthly_data.to_csv(filepath, index=False)
            print(f"Generated data for {year}-{month:02d} saved to {filename}")

# Example usage
if __name__ == "__main__":
    # Create simulator instance for a family of 4
    simulator = LaosWaterUsageSimulator(household_size=3)
    
    # Generate data for the current year
    current_year = datetime.date.today().year
    simulator.generate_yearly_data(current_year)