from agents.planner_agent import PlannerAgent

if __name__ == "__main__":
    planner = PlannerAgent()
    # user_input = input("Where would you like to go? ")

    initital_prompty = "You are an international travel planner agent. Plan a trip to {destination} from {start_date} to {end_date} for {people_num} people. The estimated budget for the trip is {budget} (optional, if known). Recommend attractions and create an itinerary, prioritizing activities from the following list: [{activity_list}]. Please also consider their preferred travel pace (e.g., fast-paced, relaxed) and accommodation style (e.g., luxury, boutique, budget) if provided. Provide a detailed breakdown of estimated costs, including categories such as accommodation, international and local transportation, food and dining, and entrance fees/activity costs. If possible, specify the cost per person where applicable."
    user_input = """
    You are an international travel planner agent. Plan a trip to Singapore from 19 July 2025 to 24 July 2025 for 2 people. The estimated budget for the trip is 3000USD (optional, if known). Recommend attractions and create an itinerary, prioritizing activities from the following list: ['sightseeing', 'shopping', 'festivals' ]. Please also consider their preferred travel pace (e.g., fast-paced, relaxed) and accommodation style (e.g., luxury, boutique, budget) if provided.
    Provide a detailed breakdown of estimated costs, including categories such as accommodation, international and local transportation, food and dining, and entrance fees/activity costs. If possible, specify the cost per person where applicable. Give the response in a json format, for example: '{_sample_prompt_json}'
    """
    _sample_prompt_json = {
        'trip_summary': {
            'destination': 'Singapore',
            'dates': {
            'start': '2025-07-19',
            'end': '2025-07-24'
            },
            'duration_days': 6,
            'travelers': 2,
            'budget_usd': 3000,
            'travel_pace': 'balanced',
            'accommodation_style': 'boutique',
            'priority_activities': ['sightseeing', 'shopping', 'festivals']
        },
        'estimated_costs': {
            'flights': 1000,
            'accommodation': 700,
            'local_transportation': 80,
            'food_and_dining': 300,
            'activities_and_attractions': 450,
            'shopping_and_misc': 300,
            'total_estimated': 2830
        },
        'accommodation_options': [
            {
            'name': 'Hotel G',
            'location': 'Dhoby Ghaut',
            'price_per_night_usd': 130
            },
            {
            'name': 'The Scarlet',
            'location': 'Chinatown',
            'price_per_night_usd': 140
            },
            {
            'name': 'Quincy Hotel',
            'location': 'Orchard',
            'price_per_night_usd': 150
            }
        ],
        'daily_itinerary': [
            {
            'day': 1,
            'date': '2025-07-19',
            'title': 'Arrival + Marina Bay Area',
            'activities': [
                'Arrive at Changi Airport',
                'Purchase EZ-Link transport card',
                'Check-in to hotel',
                'Visit Marina Bay Sands SkyPark (20 USD per person)',
                'Explore Gardens by the Bay - Supertree Grove and Cloud Forest (28 USD per person)',
                'Watch Spectra Light Show (Free)',
                'Dinner at Satay by the Bay (~20 USD total)'
            ]
            },
            {
            'day': 2,
            'date': '2025-07-20',
            'title': 'Civic District + Shopping',
            'activities': [
                'Morning walk: Singapore River, Merlion Park',
                'Visit National Gallery Singapore (15 USD per person)',
                'Lunch at Clarke Quay (~15 USD per person)',
                'Shopping at Orchard Road: ION, TANGS, Paragon',
                'Dinner at Newton Food Centre (~20 USD total)',
                'Optional rooftop bar visit (~30 USD for 2 drinks)'
            ]
            },
            {
            'day': 3,
            'date': '2025-07-21',
            'title': 'Festival & Culture',
            'activities': [
                'Visit Little India: Tekka Market, Sri Veeramakaliamman Temple',
                'Breakfast at Indian hawker (~10 USD total)',
                'Explore Kampong Glam: Sultan Mosque, Haji Lane',
                'Lunch at food festival (Singapore Food Festival – varies)',
                'Evening shopping and dinner at Bugis Street (~25 USD total)'
            ]
            },
            {
            'day': 4,
            'date': '2025-07-22',
            'title': 'Sentosa Island Day',
            'activities': [
                'Cable car to Sentosa (25 USD per person)',
                'Relax at Siloso Beach, visit Fort Siloso (Free)',
                'Lunch at Coastes (~25 USD per person)',
                'Visit S.E.A. Aquarium or Madame Tussauds (30 USD per person)',
                'Watch Wings of Time show (18 USD per person)',
                'Dinner near VivoCity (~20 USD total)'
            ]
            },
            {
            'day': 5,
            'date': '2025-07-23',
            'title': 'Nature + Shopping',
            'activities': [
                'Visit Singapore Botanic Gardens (Free)',
                'Breakfast picnic or dine-in (~15 USD total)',
                'Lunch at Dempsey Hill or Holland Village (~25 USD total)',
                'Optional: Spa or shopping (~50 USD)',
                'Dinner in Chinatown (~25 USD)',
                'Try Chili Crab (~60 USD total)'
            ]
            },
            {
            'day': 6,
            'date': '2025-07-24',
            'title': 'Departure',
            'activities': [
                'Breakfast',
                'Optional visit to Jewel Changi Airport: Rain Vortex and shopping',
                'Flight back'
            ]
            }
        ],
        'transportation': {
            'local_transport_options': [
            'EZ-Link Card or NETS FlashPay (~10-15 USD credit)',
            'MRT and Bus – extensive and reliable',
            'Grab app for short-distance rides'
            ]
        },
        'weather_and_tips': {
            'climate': 'Hot and humid in July',
            'tips': [
            'Wear breathable clothing',
            'Carry water to stay hydrated',
            'July coincides with the Great Singapore Sale and Singapore Food Festival'
            ]
        }
    }
    refined_prompty = user_input.replace('{_sample_prompt_json}', str(_sample_prompt_json))
    print(refined_prompty)
    plan = planner.run(user_input)
    print("\nFinal Trip Plan:\n")
    print(plan)