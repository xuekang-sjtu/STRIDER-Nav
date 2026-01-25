from typing import Dict
from copy import deepcopy

import numpy as np
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps as habitat_maps
from habitat.utils.visualizations.utils import draw_collision

from habitat_extensions import maps

cv2 = try_cv2_import()


def observations_to_image(observation: Dict, info: Dict, vis_info=None, items=['rgb','depth','collisions','td'], navi_area=None) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    observation_size = -1
    if "rgb" in observation and "rgb" in items:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view.append(rgb)

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation and "depth" in items:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(depth_map)

    # assert (
    #     len(egocentric_view) > 0
    # ), "Expected at least one visual sensor enabled."
    if len(egocentric_view) > 0:
        egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"] is not None and info["collisions"]["is_collision"] and "collisions" in items:
        egocentric_view = draw_collision(egocentric_view)

    # frame = egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info and "td" in items:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info and "td" in items:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]
        
        if vis_info is not None:
            if 'nodes' in vis_info:
                for p in vis_info['nodes']:
                    maps.draw_waypoint(td_map, [p[0], p[2]], info[map_k]["meters_per_px"], info[map_k]["bounds"], maps.NODE)
            if 'my_wp' in vis_info:
                for p in vis_info['my_wp']:
                    maps.draw_waypoint(td_map, [p[0], p[2]], info[map_k]["meters_per_px"], info[map_k]["bounds"], maps.DELETE)
            if 'delete' in vis_info:
                for p in vis_info['delete']:
                    maps.draw_waypoint(td_map, [p[0], p[2]], info[map_k]["meters_per_px"], info[map_k]["bounds"], maps.DELETE)
            if 'waypoints' in vis_info:
                for p in vis_info['waypoints']:
                    maps.draw_waypoint(td_map, [p[0], p[2]], info[map_k]["meters_per_px"], info[map_k]["bounds"], maps.GHOST)

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
            # if 'ghosts' in vis_info:
            #     for p in vis_info['ghosts']:
            #         maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.GHOST)
            # # if 'teacher_ghost' in vis_info and vis_info['teacher_ghost'] is not None:
            # #     maps.draw_waypoint(top_down_map, vis_info['teacher_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.TEACHER_GHOST)
            # if 'predict_ghost' in vis_info:
            #     maps.draw_waypoint(top_down_map, vis_info['predict_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.PREDICT_GHOST)
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"],
            agent_rotation=info[map_k]["agent_angle"],
            agent_radius_px=min(td_map.shape[0:2]) // 36,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        if len(egocentric_view) == 0:
            observation_size = old_h
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        # concat
        if len(egocentric_view) > 0:
            frame = np.concatenate((egocentric_view, td_map), axis=1)
        else:
            frame = td_map
    return frame

# def planner_video_frame(
#     observations,
#     info,
#     vis_info=None,
#     map_k="top_down_map_vlnce",
# ):
#     cube = {uuid: observations.pop(uuid) for uuid in UUIDS_EQ}
#     cube = {k: torch.from_numpy(v).unsqueeze(0) for k,v in cube.items()}
#     eq = CUBE2EQ(cube)
#     rgb = eq['rgbback'][0].numpy().copy()

#     top_down_map = colorize_draw_agent_and_fit_to_height(
#         info[map_k], 
#         rgb.shape[0], 
#         vis_info,
#     )
#     frame = np.concatenate([rgb, top_down_map], axis=1)
#     frame = cv2.copyMakeBorder(frame, 2,2,2,2, cv2.BORDER_CONSTANT, value=(0,0,0))
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     # frame = append_text_to_image(frame, observations["instruction"]["text"])

#     return frame


def colorize_draw_agent_and_fit_to_height(
    info: Dict, 
    output_height: int,
    vis_info: Dict = None,
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    # def _ang_dis_to_coord(
    #     ang, dis, current_position, current_heading
    # ):
    #     phi = (heading_from_quaternion(current_heading) + ang) % (2 * np.pi)
    #     x = current_position[0] - dis * np.sin(phi)
    #     z = current_position[-1] - dis * np.cos(phi)
    #     return [x, z]

    top_down_map = deepcopy(info["map"])

    if vis_info is not None:
        if 'nodes' in vis_info:
            for p in vis_info['nodes']:
                maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.NODE)
        if 'ghosts' in vis_info:
            for p in vis_info['ghosts']:
                maps.draw_waypoint(top_down_map, p[[0,2]], info["meters_per_px"], info["bounds"], maps.GHOST)
        # if 'teacher_ghost' in vis_info and vis_info['teacher_ghost'] is not None:
        #     maps.draw_waypoint(top_down_map, vis_info['teacher_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.TEACHER_GHOST)
        if 'predict_ghost' in vis_info:
            maps.draw_waypoint(top_down_map, vis_info['predict_ghost'][[0,2]], info["meters_per_px"], info["bounds"], maps.PREDICT_GHOST)
        
    top_down_map = maps.colorize_topdown_map(
        top_down_map, info["fog_of_war_mask"]
    )
    map_agent_pos = info["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info["agent_angle"],
        agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map