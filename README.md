# Poikilingo
With a personalized study plan, Poikilingo works on developing language skills and cultural awareness for bilingual children, including immigrant and refugee children, who need to learn the language of the host country as fast as possible to be able to start in the school system. In this challenge the goal will be to build a machine learning model that adapts to that data and the children’s progress in the applications, ensuring a personalized learning experience for each individual child.

## Project Goals
- Build a personalized Recommendation system where the child’s study plan adapts to the level of the child as she progresses in the plan by collecting, analyzing, and inspecting the data.
- The system also will also recommend a plan based on profile similarities (ex. Children who are learning the same language and coming from the same culture could also share similar plans or activities.)
- The model synchronizes with the child’s progress within a given interval to improve the machine learning model. Ex. After 10 activities, recommend the next activity for the child. (the partner would want this “interval” to be easily updated by their team)
- Training the model to identify the child’s next steps based on the child’s progress in the study plan. A custom dashboard to analyze, collect, inspect, etc., all the data users (inputs), all the machine learning data, and the outputs. 

## Scope
- Synchronization with learning data, perhaps every 15 days to improve machine learning.
- As mentioned above, Synchronization after X number of activities with training data to know the next steps to the user. It would be good if the number of activities could be changed in the dashboard. 
- GDPR alignment. 
- A custom dashboard to analyse, collect, inspect, etc., all the data users (inputs), all the machine learning data and the outputs. 

## Limitations:
- Since the app hasn’t been developed yet, there is no data available.
- Still the partner is working on providing dummy data that will include the user profile after a certain number of activities. This will be a JSON file.

## Additional Useful Information & Use Cases

### Recommendation System:
Each activity has important labels (should be able to add labels in each categories in the dashboard)
- Age: #2yo #3yo #4yo #5yo #6yo
- Content: #family #colors #numbers #hygiene (...)
- Cultures: #none #brazil #denmark #spain #usa (...)
- Type: #standard #sel #cultural
- GameType: #flashcards #puzzle #simulation #readingbook #coloringbook (...)
- Activity_Level: #zero #one #two

The initial course plan is fixed based on the child’s age and with activities of Activity_Level = #one
- 2yo: [ A2yo-1, A2yo-2, A2yo-3, A2yo-4 (...) 10 itens ]
- 3yo: [ A3yo-1, A3yo-2, A3yo-3, A3yo-4 (...) 10 itens ]

All recommended activities have to be picked from the pool of activities with same age group, and they have to consider the profile of the user, to keep this distribution:
- HL1 = 30-40% standard activities + 25-35% SEL activities + 35% cultural activities
- HL2 = 20-30% standard activities + 25-35% SEL activities + 45% cultural activities 
- IL1 = 15-25% standard activities + 35-45% SEL activities + 30-50% cultural activities 
- IL2 = 35-45% standard activities + 30-50% SEL activities + 15-25% cultural activities
- BTB = 50% standard activities + 20% SEL activities + 30% cultural activities

The activities Type = #cultural should be selected only if Activities’ Cultures have the same tags as Student_Info’s CulturalHeritage. This is to make sure that this specific cultural activity has content for the Cultural Heritage of the learner. 

### Examples:
- For specific examples, follow the [google drive article](https://docs.google.com/document/d/12efmKF4M_13zL26yx153fzY-r-XDlCEkPhfGmjDi1KI/edit)
