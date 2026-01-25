import re
import random
from vlnce_baselines.common.navigator.api import *
from vlnce_baselines.common.navigator.prompts import *

class Open_Nav():
    def __init__(self, device, llm_type, llm_api_key, vlm_type, vlm_api_key):
        self.device = device
        self.llm = llmClient(llm_type, llm_api_key)
        # self.spatial = spatialClient(self.device)
        # self.qwen = QwenClient(self.device)
        self.qwen = QwenAPIClient(vlm_type, vlm_api_key)
        
    # =====================================
    # ===== Instruction Comprehension =====
    # =====================================
    def get_actions(self, instruction):
        return self.llm.gpt_infer(ACTION_DETECTION['system'], ACTION_DETECTION['user'].format(instruction))

    def get_landmarks(self, actions):
        actions = actions.replace("\n", " ")
        return self.llm.gpt_infer(LANDMARK_DETECTION['system'], LANDMARK_DETECTION['user'].format(actions))
    
    # =============================
    # ===== Visual Perception =====
    # =============================
    def observe_environment(self, logger, current_step, images_list):        
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.spatial.observe_view(logger, current_step, direction_idx, direction_image)
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict
    
    # =============================
    # ===== Layour Estimation =====
    # =============================
    def observe_layout(self, logger, current_step, concat_rgb):
        observe_results = self.qwen.get_response(
            OBSERVATION_LAYOUT['system'], OBSERVATION_LAYOUT['user'], concat_rgb,
        )
        observe_results = observe_results.replace('\n', ' ')
        observe_results = f'Current Step: {current_step} Scene Description: ' + observe_results
        logger.info(observe_results)
        return observe_results
    
    # =============================
    # ===== Location Check =====
    # =============================
    def observe_location(self, logger, current_step, layout):
        observe_results = self.llm.gpt_infer(
            LOCATION_CHECK['system'], LOCATION_CHECK['user'].format(layout),
        )
        observe_results = observe_results.replace('\n', ' ')
        observe_results = f'Current Step: {current_step} Location: ' + observe_results
        logger.info(observe_results)
        return observe_results
    
    # =============================
    # ===== Single View =====
    # =============================
    def observe_waypoint(self, logger, current_step, description, images_list):
        observe_results = []
        observe_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.qwen.get_response(
            OBSERVATION_WAYPOINT['system'], OBSERVATION_WAYPOINT['user'].format(description), direction_image['rgb'],
            )
            observe_result = observe_result.replace('\n', ' ')
            observe_result = f"Direction ID: {direction_idx}, Direction: {DIRECTIONS[int(direction_idx)]}, Direction Viewpoint ID: {direction_idx}, in Step ID: {current_step}, Elevation: Eye Level, " + observe_result
            logger.info(observe_result)
            observe_results.append(observe_result) 
            observe_dict[direction_idx] = observe_result
        return observe_results, observe_dict

    # =============================
    # ===== Navigable Estimate =====
    # =============================
    def estimate_navigability(self, logger, images_list):
        # observe_results = []
        navigability_dict = {}
        for direction_idx, direction_image in images_list.items(): 
            observe_result = self.qwen.get_response(
            NAVIGABILITY_ESTIMATION['system'], NAVIGABILITY_ESTIMATION['user'], direction_image['rgb'],
            )
            observe_result = observe_result.replace('\n', '')
            observe_result = f"Direction {direction_idx}" + observe_result + ' '
            logger.info(observe_result)
            if "Score" in observe_result:
                if "Score:" in observe_result:
                    navigability_dict[direction_idx] = observe_result.split("Score:")[1].strip()
                else:
                    navigability_dict[direction_idx] =  observe_result.split("Score")[1].strip()
            else:
                navigability_dict[direction_idx] =  observe_result
        return navigability_dict
    
    # def estimate_navigability(self, logger, concat_rgb):
    #     # observe_results = []
    #     navigability_dict = {}
    #     observe_result = self.qwen.get_response(
    #     NAVIGABILITY_ESTIMATION['system'], NAVIGABILITY_ESTIMATION['user'], concat_rgb,
    #     )
    #     observe_result = observe_result.replace('\n', '')
    #     logger.info(observe_result)

    #     return navigability_dict
    
    # ===================================
    # ===== Progress Estimation =========
    # ===================================
    def save_history(self, logger, current_step, next_vp, thought, curr_observe, nav_history, executed=None): 
        # ===== get obervation summary =====
        # direction_id = int(curr_observe.split("Direction Viewpoint")[0].replace("Direction","").strip())
        direction_id = int(curr_observe.split(' ')[2][:-1])
        direction = DIRECTIONS[direction_id]
        curr_observe = "Scene Description: "+curr_observe.split("Elevation: Eye Level")[1].strip(',').strip(' ')
        observation = f"Direction {direction} " + self.llm.gpt_infer(OBSERVATION_SUMMARY['system'], OBSERVATION_SUMMARY['user'].format(curr_observe))
        # ===== get thought summary =====
        thought = self.llm.gpt_infer(THOUGHT_SUMMARY['system'], THOUGHT_SUMMARY['user'].format(thought))
        # ===== get nav history =====
        if executed is not None:
            nav_history.append({
                "step": current_step,
                "viewpoint": next_vp,
                "observation": observation,
                "thought": thought,
                "action": executed,
            })
        else:
            nav_history.append({
                "step": current_step,
                "viewpoint": next_vp,
                "observation": observation,
                "thought": thought,
            })
        logger.info(f"The history at current step is {nav_history}")
        return nav_history
    
    def action_completion(self, logger, action, movement_image): 
        response = self.qwen.get_response(
            CLOSED_COMPLETION_ESTIMATION['system'], CLOSED_COMPLETION_ESTIMATION['user'].format(action), movement_image,
        )
        response = response.replace('\n', ' ')
        # executed = response.split('Thought: ')[1].split('Executed: ')[1].strip()
        response = f'Action: {action}, ' + response
        logger.info(response)
        flag = 'true' in response.lower()
        executed = f'Action: {action}, Executed: {flag}'
        return executed, flag
    
    def obs_history(self, logger, current_step, obs_history):
        obs = self.llm.gpt_infer(OBSERVATION_HISTORY['system'], OBSERVATION_HISTORY['user'].format(current_step, obs_history))
        logger.info(obs)
        return obs

    
    def review_history(self, logger, nav_history):
        nav_history_str = " -> ".join(["Step "+str(idx+1)+" Observation: "+item["observation"]+" Thought: "+item["thought"] for idx, item in enumerate(nav_history)])
        logger.info("History: " + nav_history_str)
        return nav_history_str
    
    def estimate_completion(self, logger, actions, landmarks, history_traj):
        response = self.llm.gpt_infer(COMPLETION_ESTIMATION['system'], COMPLETION_ESTIMATION['user'].format(history_traj, landmarks, actions))
        if "Executed Actions" in response:
            logger.info("Executed Actions " + response)
            if "Executed Actions:" in response:
                return response.split("Executed Actions:")[1].strip()
            else:
                return response.split("Executed Actions")[1].strip()
        else:
            return response
    
    # =================================
    # ===== Move to next position =====
    # =================================
    def move_to_next_vp(self, logger, current_step, instruction, actions, landmarks, history_traj, estimation, layout, observation, observe_dict):    
        break_flag = True
        for i in range(1): # retry twice
            effective_prediction, thought_list = [], []
            batch_responses = self.llm.gpt_infer(NAVIGATOR['system'], 
                                                  NAVIGATOR['user'].format(observe_dict.keys(), current_step, instruction,
                                                                           actions, landmarks, history_traj, estimation, layout, observation),
                                                  num_output=3)
            if isinstance(batch_responses, str):
                batch_responses = [batch_responses]
            for decision_reasoning in batch_responses:
                logger.info(decision_reasoning)
                if "Prediction:" not in decision_reasoning:
                    continue
                logger.info(f"================retry id {i} in pred_vp==========")
                logger.info(decision_reasoning)
                pred_thought = decision_reasoning.split("Prediction:")[0].strip()
                pred_vp = decision_reasoning.split("Prediction:")[1].strip().replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")
                effective_prediction.append(pred_vp)
                thought_list.append(pred_thought)
        return effective_prediction, thought_list, break_flag
    
    # =================================
    # ===== Stop Estimation =====
    # =================================
    def stop_estimation(self, logger, instruction, actions, landmarks, history_traj, estimation, observation):    
        responses = self.llm.gpt_infer(STOP_ESTIMATION['system'], 
                                                STOP_ESTIMATION['user'].format(instruction, actions, landmarks, history_traj, estimation, observation),
                                            )
        pred_thought = responses.split("Stop:")[0].strip()
        pred_stop = responses.split("Stop:")[1].strip().replace("\"","").replace("'","").replace("\n","").replace(".","").replace("*","")
        logger.info("Stop Thought:" + pred_thought)
        logger.info("Stop:" + pred_stop)

        return pred_stop, pred_thought
    
    # =========================
    # ===== Test Decision =====
    # =========================
    def thought_fusion(self, logger, predictions, thoughts):
        matched_dict = dict()
        for pred, thought in zip(predictions, thoughts):
            if pred not in matched_dict.keys():
                matched_dict[pred] = []
            matched_dict[pred].append(thought)
            
        for key, value in matched_dict.items():
            multiple_thoughts = "; ".join(["Thought "+str(idx+1)+": "+thought for idx, thought in enumerate(value)])
            one_thought = self.llm.gpt_infer(THOUGHT_FUSION['system'], THOUGHT_FUSION['user'].format(multiple_thoughts))
            logger.info(f"Pred viewpoint ID: {key} Fused Thought: {one_thought}")
            matched_dict[key] = one_thought 
        return matched_dict 
    
    def test_decisions(self, logger, fused_pred_thought, observation, instruction, error_number, observe_dict):
        try:
            for fused_key in list(fused_pred_thought.keys()):
                if len(fused_key) > 2:
                    fused_pred_thought.pop(fused_key)
                    
            if not fused_pred_thought:
                raise ValueError("Error in fused_thought key")
                
            if len(fused_pred_thought.keys()) == 1:
                for key, value in fused_pred_thought.items():
                    return key, value, error_number
            else:
                fused_pred_thought_ = "; ".join(["Direction Viewpoint ID: "+key+" Thought: "+value for key, value in fused_pred_thought.items()])
                for i in range(2): 
                    logger.info(f"========== {i} retry in test decision==========")
                    next_vp = self.llm.gpt_infer(DECISION_TEST['system'], DECISION_TEST['user'].format(fused_pred_thought.keys(), observation, instruction, fused_pred_thought_))
                    logger.info(f"Next predicted action is {next_vp}")
                    if re.search(r'\D', next_vp):
                        next_vp = re.search(r'\d+', next_vp).group() 
        
            logger.info(f"In test decision the predicted direction: {next_vp}")
            logger.info(f"In test decision the predicted thought: {fused_pred_thought[next_vp]}")
            return next_vp, fused_pred_thought[next_vp], error_number
        except Exception as e:
            logger.info(f"Error in test decision {e}")
            error_number += 1
            logger.info(f"Error number is {error_number}")
            
            if error_number >= 2: 
                error_number = 0 
                if fused_pred_thought and all(len(key) < 2 for key in fused_pred_thought):
                    logger.info(f"Random choice a next predicted action {next_vp} in fused_pred_thought, error number reset to {error_number}")
                    next_vp, _ = random.choice(list(fused_pred_thought.items()))
                    return next_vp, fused_pred_thought[next_vp], error_number
                else:
                    next_vp, observe_description = random.choice(list(observe_dict.items()))
                    logger.info(f"Random choice a next predicted action {next_vp}, error number reset to {error_number}")
                    return next_vp, observe_description, error_number
            return "error_next_vp", "None", error_number
        


