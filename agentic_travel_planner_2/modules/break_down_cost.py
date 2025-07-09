import math
"""
Here's a breakdown of the algorithm:

Input: Total Travel Budget (e.g., $2000)

Output: A detailed travel plan with estimated costs for each component, ensuring the total cost is within the budget.

Algorithm: Travel Plan with Budget Breakdown
Phase 1: Initial Allocation & High-Level Planning

Categorize Major Expenses:

Action: Divide your total budget into broad categories. This is an initial guesstimate based on typical travel expenses.

Common Categories:

Transportation (flights, trains, buses, car rental)

Accommodation (hotels, hostels, Airbnb)

Food & Drink (restaurants, groceries, snacks)

Activities & Entertainment (tours, museums, shows)

Miscellaneous/Buffer (shopping, emergencies, unexpected costs)

Example Allocation (Initial %):

Transportation: 30-40%

Accommodation: 25-35%

Food & Drink: 15-20%

Activities: 10-15%

Miscellaneous/Buffer: 5-10%

Calculations: Multiply your total budget by these percentages to get initial dollar amounts for each category.

Example: If budget is $2000:

Transportation: $2000 * 0.35 = $700

Accommodation: $2000 * 0.30 = $600

Food & Drink: $2000 * 0.15 = $300

Activities: $2000 * 0.10 = $200

Miscellaneous: $2000 * 0.10 = $200

Define Trip Parameters (Preliminary):

Action: Based on your initial budget and preferences, roughly determine:

Destination(s): Where do you want to go? (e.g., "Southeast Asia," "European City Break," "Beach Vacation")

Duration: How long do you plan to travel? (e.g., 7 days, 14 days)

Travel Style: What kind of traveler are you? (e.g., budget-conscious, mid-range, luxury)

Constraint: Ensure these initial parameters are broadly aligned with your initial budget allocation. A luxury trip to Switzerland for a month on $2000 is likely unrealistic.

Phase 2: Detailed Research & Refinement (Iterative Process)

Research Transportation Costs:

Action: Start with the most significant transportation cost, usually flights if traveling internationally.

Steps:

Use flight comparison websites (Google Flights, Skyscanner, Kayak) to get realistic estimates for your chosen destination(s) and travel dates (be flexible if possible).

Consider ground transportation (trains, buses, car rental) for within-country travel.

Update Budget: Adjust your "Transportation" category based on actual flight estimates. If flights are much higher than your initial allocation, you'll need to re-evaluate other categories or your destination/duration.

Research Accommodation Costs:

Action: Research typical accommodation prices for your chosen destination(s) and travel style.

Steps:

Use booking sites (Booking.com, Airbnb, Hostelworld) to find average nightly rates for hotels, hostels, guesthouses, or apartments.

Factor in the number of nights.

Update Budget: Adjust your "Accommodation" category. If it's too high, consider cheaper options (hostels, guesthouses, staying further from city centers) or reducing trip duration.

Estimate Food & Drink Costs:

Action: Research average daily food costs for your destination.

Steps:

Look up typical meal prices (breakfast, lunch, dinner) at different types of establishments (local eateries vs. tourist restaurants).

Consider if you'll cook some meals (e.g., if staying in an Airbnb with a kitchen).

Multiply the daily estimate by the number of days.

Update Budget: Adjust your "Food & Drink" category.

Estimate Activities & Entertainment Costs:

Action: Identify potential activities, tours, and attractions you want to experience.

Steps:

Research entrance fees for museums, parks, attractions.

Look up prices for tours (e.g., walking tours, day trips).

Allocate a budget for spontaneous activities or souvenirs.

Update Budget: Adjust your "Activities" category. Prioritize "must-do" activities.

Calculate Miscellaneous/Buffer:

Action: Keep this category for unexpected costs, souvenirs, local transport (taxis, public transport), visas, travel insurance, etc.

Rule of Thumb: Aim for 5-10% of your total budget, especially for international travel. This is crucial for avoiding budget overruns.

Phase 3: Review & Optimization (Continuous Adjustment)

Calculate Current Total Cost:

Action: Sum up the current estimated costs for all categories (Transportation + Accommodation + Food & Drink + Activities + Miscellaneous).

Compare to Total Budget:

Decision Point:

If Current Total Cost <= Total Budget: You're good! Proceed to finalize your plan. You might even have a small surplus.

If Current Total Cost > Total Budget: This is where the iterative part comes in. You need to make adjustments.

Adjust & Optimize (Iterate):

Action: Go back to the categories and look for areas to cut costs. Prioritize areas with the largest discrepancies.

Strategies:

Reduce Trip Duration: Fewer days mean less accommodation, food, and activities.

Change Destination: A cheaper country/city.

Adjust Travel Style: Opt for cheaper accommodation (hostels, guesthouses), less expensive food options (street food, cooking), free activities.

Be Flexible with Dates: Traveling during off-peak seasons can significantly reduce flight and accommodation costs.

Look for Deals: Early bird specials, package deals, last-minute offers (with caution).

Self-Cater More: Cook more meals if accommodation allows.

Prioritize Activities: Cut non-essential activities.

Cheaper Transportation: Bus instead of train, public transport instead of taxis.

Repeat Steps 3-9: Continue adjusting and recalculating until your total estimated cost is within your budget.

Phase 4: Finalization & Detailed Breakdown

Create Detailed Breakdown:

Action: Once the overall plan fits the budget, break down each major category into specific line items.

Example:

Transportation:

Flights (Round Trip): $XXX

Train from A to B: $XX

Local Metro Pass (7 days): $XX

Airport Transfer: $XX

Accommodation:

Hotel A (3 nights @ $XX/night): $XXX

Hostel B (4 nights @ $XX/night): $XXX

Food & Drink (per day estimate x days):

Breakfast: $X/day * 7 days = $XX

Lunch: $X/day * 7 days = $XX

Dinner: $X/day * 7 days = $XX

Snacks/Drinks: $X/day * 7 days = $XX

Activities:

Museum Entrance: $XX

Guided Tour: $XX

Concert Ticket: $XX

Miscellaneous:

Travel Insurance: $XX

Visa Fee: $XX

Souvenirs: $XX

Emergency Fund: $XX

Contingency Planning:

Action: Always aim to have a small buffer (part of your "Miscellaneous" category) for unforeseen circumstances or unexpected opportunities.

Key Principles of this Algorithm:

Iterative: You will go back and forth between planning and budgeting until you achieve a satisfactory result.

Top-Down then Bottom-Up: Start with high-level budget allocation, then drill down into specific costs.

Research-Driven: Accurate estimates rely heavily on good research.

Flexibility: Be prepared to compromise on some aspects of your trip (destination, duration, style) to fit the budget.

Prioritization: Decide what aspects of the trip are most important to you and allocate funds accordingly.

This algorithmic approach provides a robust framework for planning a travel budget, even when starting with just a total budget.
"""
def get_float_input(prompt):
    """Helper function to get a valid float input."""
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                print("Value must be positive. Please try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_int_input(prompt):
    """Helper function to get a valid integer input."""
    while True:
        try:
            value = int(input(prompt))
            if value <= 0:
                print("Value must be positive. Please try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter an integer.")

def get_yes_no_input(prompt):
    """Helper function for yes/no questions."""
    while True:
        response = input(prompt).lower().strip()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

def get_choice_input(prompt, choices):
    """Helper function for numbered choices."""
    while True:
        print(prompt)
        for i, choice in enumerate(choices):
            print(f"{i+1}. {choice}")
        try:
            choice_num = int(input("Enter your choice number: "))
            if 1 <= choice_num <= len(choices):
                return choices[choice_num - 1]
            else:
                print("Invalid choice number. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def generate_travel_plan():
    print("--- Travel Plan Generator with Budget Breakdown ---")
    print("Let's create your travel plan based on your total budget.")

    total_budget = get_float_input("Enter your total travel budget (e.g., 2000): $")
    num_travelers = get_int_input("How many people are traveling? ")

    print("\n--- Phase 1: Initial Allocation & High-Level Planning ---")

    # Define estimated cost profiles for different destination types (per person, per day)
    # These are very rough estimates and would be replaced by real data in a true system.
    cost_profiles = {
        "Budget-Friendly (e.g., Southeast Asia, Eastern Europe)": {
            "flight_multiplier": 0.35, # % of budget for flights
            "daily_accommodation": 25, # USD per person
            "daily_food": 20,         # USD per person
            "daily_activities": 15    # USD per person
        },
        "Mid-Range (e.g., Western Europe, North America)": {
            "flight_multiplier": 0.40,
            "daily_accommodation": 60,
            "daily_food": 45,
            "daily_activities": 30
        },
        "High-End (e.g., Switzerland, Japan, Luxury Resorts)": {
            "flight_multiplier": 0.45,
            "daily_accommodation": 150,
            "daily_food": 80,
            "daily_activities": 60
        }
    }

    destination_style_choices = list(cost_profiles.keys())
    selected_style_name = get_choice_input("\nWhat kind of destination/travel style are you aiming for?", destination_style_choices)
    selected_profile = cost_profiles[selected_style_name]

    # Initial trip parameters
    trip_duration_days = get_int_input(f"How many days do you plan to travel? (e.g., 7, 14): ")
    print("\n")

    # Initial Allocation Percentages (can be adjusted by user later in iteration)
    # These are generalized and will be refined by the cost profiles
    allocation_percentages = {
        "transportation": 0.35,
        "accommodation": 0.30,
        "food_drink": 0.15,
        "activities": 0.10,
        "miscellaneous": 0.10 # Buffer, shopping, local transport, visa, etc.
    }

    # Initialize costs (will be updated)
    estimated_costs = {
        "transportation": 0.0,
        "accommodation": 0.0,
        "food_drink": 0.0,
        "activities": 0.0,
        "miscellaneous": total_budget * allocation_percentages["miscellaneous"]
    }

    def calculate_current_costs(profile, duration, travelers, budget):
        # Transportation (initial estimate based on a percentage of the total budget for flights)
        # This is very rough for initial planning
        trans_cost = budget * profile["flight_multiplier"]

        # Accommodation (per person per night)
        acc_cost = profile["daily_accommodation"] * duration * travelers

        # Food & Drink (per person per day)
        food_cost = profile["daily_food"] * duration * travelers

        # Activities (per person per day)
        act_cost = profile["daily_activities"] * duration * travelers

        # Miscellaneous is kept constant initially unless explicitly cut
        misc_cost = estimated_costs["miscellaneous"] # Use the already calculated initial misc

        return {
            "transportation": trans_cost,
            "accommodation": acc_cost,
            "food_drink": food_cost,
            "activities": act_cost,
            "miscellaneous": misc_cost
        }

    # --- Phase 2 & 3: Detailed Research & Refinement (Iterative Process) ---
    current_iteration_costs = calculate_current_costs(
        selected_profile, trip_duration_days, num_travelers, total_budget
    )
    estimated_costs.update(current_iteration_costs) # Update with the profile-based estimates

    def print_budget_summary(current_costs, budget):
        current_total = sum(current_costs.values())
        print("\n--- Current Estimated Budget ---")
        for category, cost in current_costs.items():
            print(f"- {category.replace('_', ' ').title():<15}: ${cost:,.2f}")
        print(f"{'-'*30}")
        print(f"Total Estimated Cost : ${current_total:,.2f}")
        print(f"Total Budget         : ${budget:,.2f}")
        if current_total > budget:
            print(f"OVER BUDGET by       : ${current_total - budget:,.2f}!")
        else:
            print(f"Remaining Budget     : ${budget - current_total:,.2f}")
        print(f"{'-'*30}\n")
        return current_total

    current_total_cost = print_budget_summary(estimated_costs, total_budget)

    while current_total_cost > total_budget:
        print("Your plan is currently over budget. Let's make some adjustments.")
        print("What would you like to do?")
        adjustment_choices = [
            "Reduce trip duration",
            "Change travel style/destination type",
            "Adjust specific category costs (e.g., cheaper accommodation, fewer activities)",
            "Decrease miscellaneous/buffer"
        ]
        adjustment_option = get_choice_input("Select an adjustment strategy:", adjustment_choices)

        if adjustment_option == "Reduce trip duration":
            print(f"Current duration: {trip_duration_days} days.")
            new_duration = get_int_input("Enter new shorter duration in days: ")
            if new_duration >= trip_duration_days:
                print("New duration must be shorter. No change applied.")
            else:
                trip_duration_days = new_duration
                # Recalculate daily-dependent costs based on new duration
                estimated_costs["accommodation"] = selected_profile["daily_accommodation"] * trip_duration_days * num_travelers
                estimated_costs["food_drink"] = selected_profile["daily_food"] * trip_duration_days * num_travelers
                estimated_costs["activities"] = selected_profile["daily_activities"] * trip_duration_days * num_travelers
                print(f"Trip duration updated to {trip_duration_days} days.")

        elif adjustment_option == "Change travel style/destination type":
            new_style_name = get_choice_input("Select a new (likely cheaper) travel style:", destination_style_choices)
            selected_style_name = new_style_name
            selected_profile = cost_profiles[new_style_name]
            # Recalculate all main costs based on new profile
            estimated_costs.update(calculate_current_costs(
                selected_profile, trip_duration_days, num_travelers, total_budget
            ))
            print(f"Travel style updated to '{selected_style_name}'. Costs recalculated.")

        elif adjustment_option == "Adjust specific category costs (e.g., cheaper accommodation, fewer activities)":
            category_to_adjust_choices = ["Transportation", "Accommodation", "Food_Drink", "Activities"]
            category_name_raw = get_choice_input("Which category do you want to adjust?", category_to_adjust_choices).lower()
            category_key = category_name_raw.replace(" ", "_")

            print(f"Current {category_key.replace('_', ' ').title()}: ${estimated_costs[category_key]:,.2f}")
            new_amount = get_float_input(f"Enter the new estimated cost for {category_key.replace('_', ' ').title()}: $")
            if new_amount >= estimated_costs[category_key]:
                print("New amount must be lower to save money. No change applied.")
            else:
                estimated_costs[category_key] = new_amount
                print(f"{category_key.replace('_', ' ').title()} cost updated.")

        elif adjustment_option == "Decrease miscellaneous/buffer":
            print(f"Current Miscellaneous/Buffer: ${estimated_costs['miscellaneous']:,.2f}")
            new_misc_amount = get_float_input("Enter the new amount for Miscellaneous/Buffer: $")
            if new_misc_amount >= estimated_costs['miscellaneous']:
                print("New amount must be lower to save money. No change applied.")
            else:
                estimated_costs['miscellaneous'] = new_misc_amount
                print("Miscellaneous/Buffer updated.")

        current_total_cost = print_budget_summary(estimated_costs, total_budget)

    print("\n--- Phase 4: Finalization & Detailed Breakdown ---")
    print("Great! Your plan is now within budget.")
    print("Here's your estimated travel plan breakdown:")

    # Detailed Breakdown (this part is more illustrative as actual sub-items would need more input)
    print(f"\nTrip Parameters:")
    print(f"  Total Budget: ${total_budget:,.2f}")
    print(f"  Number of Travelers: {num_travelers}")
    print(f"  Trip Duration: {trip_duration_days} days")
    print(f"  Travel Style/Destination Type: {selected_style_name}")

    print("\n--- Estimated Cost Breakdown ---")

    # Transportation Detail
    print("\n1. Transportation:")
    print(f"   Estimated Flights: ${estimated_costs['transportation']:,.2f}")
    # In a real app, this would prompt for specific flight costs, train costs, etc.
    print(f"   (Includes main flights/long-distance travel. Allocate separately for local transport.)")
    print(f"   Subtotal: ${estimated_costs['transportation']:,.2f}")

    # Accommodation Detail
    print("\n2. Accommodation:")
    avg_acc_per_night_pp = estimated_costs['accommodation'] / (trip_duration_days * num_travelers) if (trip_duration_days * num_travelers) > 0 else 0
    print(f"   Estimated total: ${estimated_costs['accommodation']:,.2f}")
    print(f"   (Roughly ${avg_acc_per_night_pp:,.2f} per person per night)")
    print(f"   Subtotal: ${estimated_costs['accommodation']:,.2f}")

    # Food & Drink Detail
    print("\n3. Food & Drink:")
    avg_food_per_day_pp = estimated_costs['food_drink'] / (trip_duration_days * num_travelers) if (trip_duration_days * num_travelers) > 0 else 0
    print(f"   Estimated total: ${estimated_costs['food_drink']:,.2f}")
    print(f"   (Roughly ${avg_food_per_day_pp:,.2f} per person per day)")
    print(f"   Subtotal: ${estimated_costs['food_drink']:,.2f}")

    # Activities & Entertainment Detail
    print("\n4. Activities & Entertainment:")
    avg_activities_per_day_pp = estimated_costs['activities'] / (trip_duration_days * num_travelers) if (trip_duration_days * num_travelers) > 0 else 0
    print(f"   Estimated total: ${estimated_costs['activities']:,.2f}")
    print(f"   (Roughly ${avg_activities_per_day_pp:,.2f} per person per day for tours, museums, etc.)")
    print(f"   Subtotal: ${estimated_costs['activities']:,.2f}")

    # Miscellaneous/Buffer Detail
    print("\n5. Miscellaneous/Buffer:")
    print(f"   Emergency fund, local transport, souvenirs, travel insurance, visa fees.")
    print(f"   Subtotal: ${estimated_costs['miscellaneous']:,.2f}")

    print(f"\n{'='*40}")
    final_total = sum(estimated_costs.values())
    print(f"FINAL TOTAL ESTIMATED COST: ${final_total:,.2f}")
    print(f"Budget Remaining: ${total_budget - final_total:,.2f}")
    print(f"{'='*40}")

    print("\n--- Next Steps ---")
    print("Now that you have a high-level plan, start researching actual prices for flights, hotels, and attractions.")
    print("Be prepared to refine these estimates further as you get concrete quotes.")
    print("Consider creating a detailed daily itinerary to allocate food and activity costs more precisely.")
    print("Have a fantastic trip!")

if __name__ == "__main__":
    generate_travel_plan()