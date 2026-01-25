# Actions Decompsition
ACTION_DETECTION = {
    'system': "You are an expert in action decomposition. Your task is to identify and list all the actions described in the given navigation instruction. \
                Break the instruction down into the smallest meaningful steps, and arrange them in the correct order of execution. \
                For example, for the instruction \"turn right after passing the table\", the correct decomposition is: \"pass the table\", \"turn right\". \
                If the instruction is logically incomplete, you should supplement it based on reasonable assumptions. \
                For example, \"exit the bathroom. exit the bedroom\" should be decomposed as: \"exit the bathroom\", \"enter the bedroom\", \"exit the bedroom\". \
                Ensure that each action is logically complete and meaningful on its own. \
                Your response must consist only of a list of labeled action phrases, without full sentences. ",
    'user': "Can you decompose actions in the instruction \"{}\"? Actions: "
}

# Landmarks Extraction
LANDMARK_DETECTION = {
    'system': "You are a landmark extraction expert. Your task is to detect all landmarks in the given navigation instruction. You need to ensure the integrity of each landmarks. Your answer must consist ONLY of a series of labled landmark phrases without other sentences.",
    'user': "Can you extract landmarks in the instruction \"{}\"? Landmarks: "
}

# Directions in Observation
DIRECTIONS = ["Front, range(left 15 to right 15)", "Front Left, range(left 15 to left 45)", "Left, range(left 45 to left 75)", "Left, range(left 75 to left 105)", "Rear Left, range(left 105 to left 135)", "Rear Left, range(left 135 to left 165)",
                    "Back, range(left 165 to right 165)", "Rear Right, range(right 135 to right 165)", "Right, range(right 105 to right 135)", "Right, range(right 75 to right 105)", "Front Right, range(right 45 to right 75)", "Front Right, range(right 15 to right 45)"]

# Summarize Observation
OBSERVATION_SUMMARY = {
    'system': "You are a trajectory summary expert. Your task is to simplify environment description as short and clear as possible. \
                                            You ONLY need to summarize in a single paragraph.",
    'user': "Given Environment Description \"{}\", Summarization:" 
}

OBSERVATION_HISTORY = {
    'system': "You are a history observation summary expert. Your task is to simplify all history environment descriptions as short and clear as possible. \
                                            You ONLY need to summarize in a single paragraph.",
    'user': "Given Current Step \"{}\", All History Environment Description \"{}\", Summarization:"
}

# Summarize Thought
THOUGHT_SUMMARY = {
    'system': "You are a trajectory summary expert. Your task is to simplify navigation thought process as short and clear as possible. \
                                            You ONLY need to summarize the what actions you did and what landmarks you passed in \"Thought\" using a single paragraph. Do NOT include Direction information. ",
    'user': "Given Thought Process \"{}\", Summarization:"
}

STOP_ESTIMATION = {
    'system': "You are a navigation agent tasked with following instructions to move through an indoor environment using the fewest possible action steps. \
           I will provide you with one instruction and a list of landmarks. You will also receive your navigation history and an estimation of executed actions for reference. \
           You can observe your current environment through scene descriptions, identified objects, and any visible landmarks in various directions around you. \
           Your goal is to assess whether you have reached the destination described in the instruction. \
           Instructions typically specify where you should stop at the end. \
           Your answer must contain two parts: \"Thought\" and \"Stop\". You must label these parts clearly, without adding any extra symbols. \
           In the \"Stop\" section, your answer must be either YES or NO, indicating whether you should stop or continue navigating. \
           In the \"Thought\" section, you must follow a structured, detailed analysis process: \
           (1) Analyze the instruction and infer the intended stopping point. \
           (2) Check whether all specified landmarks and actions have been observed in the navigation history. You must not stop unless all required landmarks have been reached. \
           (3) Examine your current observations to determine whether they match the expected stop location from the instruction. You must not stop unless the observations match the stopping point. \
           You must double-check that the output in the \"Stop\" section is either YES or NO, without any additional words. \
           You must also ensure that the \"Thought\" section consists of a single, well-structured paragraph.",
    'user': "Instruction: {} ({}) Landmarks: {} Navigation History: {} Estimation of Executed Actions: {} Current Environment: {} -> Thought: ... Stop: ... \
            Your output after \"Stop\" must be YES or NO without any other words. \
            Your output after \"Thought\" must be a single paragraph explaining why you should or should not stop."
}

# Estimate Completion
COMPLETION_ESTIMATION = {
    'system': "You are a completion estimation expert. Your task is to estimate what actions in the instruction have been executed based on navigation history and landmarks. \
                All actions in the instruction are given following the temporal order. Your answer includes two parts: \"Thought\" and \"Executed Actions\". You need to use \"Thought\" and \"Executed Actions\" without any other symbols. \
                In the \"Thought\", you must follow procedures to analyze as detailed as possible what actions have been executed: \
                (1) What given landmarks of actions have appeared in the navigation history? \
                (2) Analyze the direction change at each step in the navigation history. \
                (3) Estimate each action in the instruction based on each step in the navigation history to check their completion. \
                (4) You must estimate actions in order. This means that if action 1 is not completed, you can not completed actions 2. \
                In the \"Executed Actions\", you must only write down actions that have been executed without other words. \
                You must strictly refer original actions in the given instruction to estimate.",
    'user': "Given Navigation History \"{}\" and Landmarks in the instruction \"{}\", estimate what actions in instruction \"{}\" have been executed."
}

# Estimate Completion
CLOSED_COMPLETION_ESTIMATION = {
    'system': "You are an expert in action completion estimation. Your task is to determine whether the given action has been successfully executed. \
                You are provided with two images: the left image shows the previous view, and the right image shows the view after the agent has taken an action. \
                Please follow the step-by-step instructions provided below: \
                (1) Understand the given action and determine what observable change is expected if the action is completed. \
                (2) Compare the views and focus on changes in object positions and sizes, appearance or disappearance of certain landmarks. \
                (3) Determine completion criteria: \
                    \"Go towards X\": The target X should appear larger or more centered in the right image, indicating that the agent moved closer. \
                    \"Pass through the door\" or \"Enter room\": The door should be behind or no longer visible in the right image, and the surroundings should match the expected next area. \
                    \"Leave the room\": Room-specific objects should no longer be visible, and another room elements should start to appear. \
                    \"Turn left/right\": The view should shift accordingly. Objects on one side should move toward the center or out of view. \
                    \"Move forward\": The general scene should look closer or more zoomed-in; far objects should become more prominent. \
                Your answer includes two parts: \"Thought\" and \"Executed\". \
                In the \"Thought\", you must follow instructions to analyze whether the given action has been executed in a single paragraph. \
                In the \"Executed\", you must output \"True\" or \"False\" without any other words. ",
    'user': "Given Action \"{}\", estimate whether the given action has been successfully executed. Thought: ... Executed: ... "
}

# Main Navigator
NAVIGATOR = {
    'system': "You are a navigation agent who follows instruction to move in an indoor environment with the least action steps. \
            I will give you one instruction and tell you landmarks. I will also give you navigation history and estimation of executed actions for reference. \
            You can observe current environment by scene descriptions, scene objects and possible existing landmarks in different directions around you. \
            The Current Environment describes an overall environment of your surroundings. \
            Each direction contains direction viewpoint ids you can move to. Your task is to predict moving to which direction viewpoint. \
            In each prediction, direction 0 always represents your current orientation. Direction 1 represents the direction that is 30 degrees to the left of direction 0, Direction 2 represents the direction that is 60 degrees to the left of direction 0, Direction 3 represents the direction that is 90 degrees to the left of direction 0, Direction 4 represents the direction that is 120 degrees to the left of direction 0, Direction 5 represents the direction that is 150 degrees to the left of direction 0, Direction 6 represents the direction that is 180 degrees to the left of direction 0, Direction 7 represents the direction that is 150 degrees to the right of direction 0, Direction 8 represents the direction that is 120 degrees to the right of direction 0, Direction 9 represents the direction that is 90 degrees to the right of direction viewpoint ID 0, Direction 10 represents the direction that is 60 degrees to the right of direction 0, Direction 11 represents the direction that is 30 degrees to the right of direction 0 \
            Note that environment direction that contains more landmarks mentioned in the instruction is usually the better choice for you. \
            If you are required to go up stairs, you need to move to direction with higher position. If you are required to go down stairs, you need to move to direction with lower position. \
            You are encouraged to move to new viewpoints to explore environment while avoid revisiting accessed viewpoints in non-essential situations. \
            If you feel struggling to find the landmark or execute the action, you can try to execute the subsequent action and find the subsequent landmark. \
            Your answer includes two parts: \"Thought\" and \"Prediction\". In the \"Thought\", you should think as detailed as possible following procedures: \
            (1) The viewpoint ID you predicted must be one of the Direction Viewpoint ID in Candidate Viewpoint IDs List. The Candidate Viewpoint IDs List show the Direction Viewpoint ID that you should go. This means that there should be only a number after \"Prediction\" without any other words or characters . \
            (2) Check whether the latest executed action has been completed by comparing current environment/view and landmark in the latest executed action. \
            (3) Determine the action you should execute and landmark you should reach now. If the latest executed action have not been completed, \
                you should continue to execute it. Otherwise, you should execute the next action in the given instruction. \
            (4) Analyze which direction in the current environment/view is most suitable to execute the action you decide and explain your reason. \
            (5) Predict moving to which direction viewpoint based on your thought process. \
            (6) The \"Thought\" you predicted should be a single paragraph. \
            (7) If you believe you have completed the instruction, you must still strictly follow the requirements to predict the next viewpoint in the \"Prediction\". \
            (8) If you want to make a left turn, you usually need to select a viewpoint ID between 2 and 4. If you want to make a right turn, you usually need to select a viewpoint ID between 8 and 10. \
                If you want to turn around, you usually need to select a viewpoint ID between 5 and 7. If you want to go straight, you usually need to select a viewpoint ID among 0, 1 and 11. \
                However, the viewpoint ID you predict must be within the Current View.\
            (9) Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
            (10) If the instruction does not explicitly indicate a turn, try to maintain your current direction—that is, choose the point directly in front of you. \
            Then, please make decision on the next viewpoint in the \"Prediction\". \
            Your decision is very important, must make it very carefully. \
            You need to double check the output in \"Prediction:\". The output must be in the Candidate Viewpoint IDs without any other words. \
            You also need to double check the output in \"Thought\". The output must be a single paragraph",
    'user': "Candidate Viewpoint IDs List: [{}] Step {} Instruction: {} ({}) Landmarks: {} Navigation History: {} \
            Estimation of Executed Actions: {} Current Environment: {} Candidate View: {}-> Thought: ... Prediction: ... \
            Your output after \"Prediction\" must be one of the number in Candidate Viewpoint IDs List without any other words. \
            Your output after \"Thought\" must be a single paragraph about why you choose this viewpoint id. "
}

# Thought Fusion
THOUGHT_FUSION = {
    'system': "You are a thought fusion expert. Your task is to fuse given thought processes \
                    into one thought. You need to reserve key information related to actions, landmarks, direction changes. You should only answer fused thought without other words.",
    'user': "Can you help me fuse the thoughts leading to the same movement direction? The thoughts are :{}, Fused thought: "
}

# Test Decision
DECISION_TEST = {
    'system': "You are a decision testing expert. Your task is to evaluate the feasibility of each movement \
                        prediction based on thought process and environment. Then, you will make a final decision about direction viewpoint ID without other words. \
                            The answer should only be a number and within the candidate list.",
    'user': "The candidate list: {}. Can you help me make a final decision? The Observation: {}, Navigation Instruction: {}, {}, Final Decision: "
}

# Layout observation
# OBSERVATION_LAYOUT = {
#     'system': "You are an expert in environmental description. Your task is to describe the objects and areas visible in a panoramic image, along with their spatial relationships. \
#             The images were captured from a fixed position as the camera rotated clockwise in a full circle. \
#             Your response must include three parts: \"Objects and Areas\" and \"Positions\". \
#             In \"Objects and Areas\", describe the visible objects and distinguishable areas in a single paragraph. \
#             In \"Positions\", explain the relative positions of these objects and areas in a single paragraph. \
#             Do not output anything other than \"Objects and Areas\" and \"Positions\". ",
#     'user': "The pictures are taken from a fixed position, with the camera rotating in a circular manner. \
#             Describe the surrounding environment, including visible objects and distinct areas. Then infer their positions. Objects and Areas: ... Positions: ... "
# }
OBSERVATION_LAYOUT = {
    'system': "You are an expert in environmental description. Your task is to describe the objects and areas visible in a panoramic image, along with their spatial relationships. \
            The images were captured from a fixed position as the camera rotated clockwise in a full circle. \
            Your response must include \"Objects and Areas\". \
            In \"Objects and Areas\", describe the visible objects and distinguishable areas in a single paragraph. \
            Do not output anything other than \"Objects and Areas\"",
    'user': "The pictures are taken from a fixed position, with the camera rotating in a circular manner. \
            Describe the surrounding environment, including visible objects and distinct areas. Objects and Areas: ... "
}


# waypoint observation
OBSERVATION_WAYPOINT = {
    'system': "You are provided with a description of your current surroundings and an image taken from the same location. \
                Based on the overall description and the image, your task is to identify the objects or areas directly in front of you. \
                Please respond with a single, brief paragraph.",
    'user': "Description: {}. What objects or areas can you see? "
}
# OBSERVATION_WAYPOINT = {
#     'system': "You are given an image captured from the a curtain location. \
#                 Your task is to determine what objects or areas you are currently facing, such as bed, door, kitchen and hallway. \
#                 You need to answer in a single paragraph. ",
#     'user': "Can you tell me what objects or areas you are facing? "
# }

# location check
LOCATION_CHECK = {
    'system': "You are a location perception expert. Your task is to infer your current room or area based on the surrounding environment. \
                Given a description of your surroundings, you should summarize the type of space you are in—such as a bedroom, hallway, or intersection—using concise terms.",
    'user': "The description is as follows: {}. Can you tell me what type of space you are in??"
}

# navigability
NAVIGABILITY_ESTIMATION = {
    'system': "The agent has been tasked with navigating in an indoor environment. The agent has sent you an image of a direction describing your surrounding environment. \
            Your job is to predict whether the direction is navigable and predict a score to the direction (ranging from 0 to 10). \
            Please follow my step-by-step instructions: \
            (1) A scene with no feasible path and no target is assigned a score of 0. \
            (2) A scene which is totally navigable is assigned a score of 10. \
            (3) Intermediate cases are assigned scores ranging from 0 to 10. \
            For example, a scene with a wall or obstacle should be assigned a score of 0, while a scene with an open area should be assigned a score of 10. \
            Your answer includes two parts: \"Thought\" and \"Score\". \
            Your output after \"Thought\" must be a single paragraph about why you assign the score to the direction. \
            Your output after \"Score\" must be an interger range from 0 to 10 without any other words.",
    'user': "Please consider whether the direction is navigable and assign scores for the direction (ranging from 0 to 10)."
}

# NAVIGABILITY_ESTIMATION = {
#     'system': "The agent has been tasked with navigating in an indoor environment. The agent has sent you 12 images of different directions describing your surrounding environment. \
#             Your job is to predict whether each direction is navigable and predict a score to each direction (ranging from 0 to 10). \
#             Please follow my step-by-step instructions: \
#             (1) A scene with no feasible path and no target is assigned a score of 0. \
#             (2) A scene which is totally navigable is assigned a score of 10. \
#             (3) Intermediate cases are assigned scores ranging from 0 to 10. \
#             For example, a scene with a wall or obstacle should be assigned a score of 0, while a scene with an open area should be assigned a score of 10. \
#             Format your answer like {'Picture 1': {'Score': <The score(from 0 to 10) of Picture 1>, 'Explanation': <An explanation for your assigned score.>}, 'Picture 2': {...}, …}.",
#     'user': "Please consider whether each direction is navigable and assign scores for each direction (ranging from 0 to 10)."
# }