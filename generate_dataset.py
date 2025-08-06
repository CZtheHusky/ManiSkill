from internvl_eval.utils import ManiSkillTrajectoryDataset, InternVLPretrainDatasetGenerator

# dataset = ManiSkillTrajectoryDataset(dataset_file="StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5", success_only=False, device=None)

# generator = InternVLPretrainDatasetGenerator(
#     dataset=dataset,
#     save_path="stack_horizon",
#     horizon=1,
#     dual_camera=True,
#     scale_factor=100,
#     )
# # generator.cal_statistics()
# generator.generation()

dataset = ManiSkillTrajectoryDataset(dataset_file="StackCube-v1/motionplanning/trajectory.rgb.pd_ee_delta_pose.physx_cpu.h5", success_only=False, device=None)

generator = InternVLPretrainDatasetGenerator(
    dataset=dataset,
    save_path="stack_cubes",
    horizon=1,
    dual_camera=True,
    )
# generator.cal_statistics()
generator.generation()