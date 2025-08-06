from internvl_eval.utils import ManiSkillTrajectoryDataset, InternVLPretrainDatasetGenerator

dataset = ManiSkillTrajectoryDataset(dataset_file="StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5", success_only=False, device=None, is_episode_dataset=True)

generator = InternVLPretrainDatasetGenerator(
    dataset=dataset,
    save_path="stack_cubes_horizon",
    horizon=4,
    dual_camera=True,
    )
# generator.cal_statistics()
generator.traj_generation()