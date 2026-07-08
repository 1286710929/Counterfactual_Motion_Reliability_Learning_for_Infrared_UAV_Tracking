import os

from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    output_root = os.path.join(package_root, "output")
    data_root = os.environ.get("CMRTRACK_DATA_ROOT", "/home/cyh/Anti_UAV/datasets")

    settings.prj_dir = package_root
    settings.save_dir = output_root
    settings.results_path = os.path.join(output_root, "test/tracking_results")
    settings.segmentation_path = os.path.join(output_root, "test/segmentation_results")
    settings.network_path = os.path.join(output_root, "test/networks")
    settings.result_plot_path = os.path.join(output_root, "test/result_plots")

    settings.antiuav410_path = os.environ.get(
        "ANTIUAV410_PATH", os.path.join(data_root, "Anti-UAV410")
    )
    settings.antiuav_rgbt_path = os.environ.get(
        "ANTIUAV_RGBT_PATH", os.path.join(data_root, "Anti-UAV-RGBT")
    )

    settings.otb_path = os.path.join(package_root, "data/otb")
    settings.nfs_path = os.path.join(package_root, "data/nfs")
    settings.uav_path = os.path.join(package_root, "data/uav")
    settings.tc128_path = os.path.join(package_root, "data/TC128")
    settings.vot_path = os.path.join(package_root, "data/VOT2019")
    settings.vot18_path = os.path.join(package_root, "data/vot2018")
    settings.vot22_path = os.path.join(package_root, "data/vot2022")
    settings.got10k_path = os.path.join(package_root, "data/got10k")
    settings.got10k_lmdb_path = os.path.join(package_root, "data/got10k_lmdb")
    settings.lasot_path = os.path.join(package_root, "data/lasot")
    settings.lasot_lmdb_path = os.path.join(package_root, "data/lasot_lmdb")
    settings.trackingnet_path = os.path.join(package_root, "data/trackingnet")
    settings.itb_path = os.path.join(package_root, "data/itb")
    settings.tnl2k_path = os.path.join(package_root, "data/tnl2k")
    settings.lasot_extension_subset_path = os.path.join(package_root, "data/lasot_extension_subset")

    settings.tpl_path = ""
    settings.davis_dir = ""
    settings.youtubevos_dir = ""
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.tn_packed_results_path = ""

    return settings
