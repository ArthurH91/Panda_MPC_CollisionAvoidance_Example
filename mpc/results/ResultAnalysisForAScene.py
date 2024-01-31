from os import listdir
from os.path import dirname, join, abspath, isfile
import json, codecs

import numpy as np
import pinocchio as pin
import hppfcl
import matplotlib.pyplot as plt

from Result import Result

import seaborn as sns
plt.rcParams.update({"font.size": 22})
import numpy as np

import matplotlib as mpl
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection
import crocoddyl
import pinocchio as pin

sns.set_palette("colorblind") 
DEFAULT_FONT_SIZE = 25
DEFAULT_AXIS_FONT_SIZE = DEFAULT_FONT_SIZE
DEFAULT_LINE_WIDTH = 4  # 13
DEFAULT_MARKER_SIZE = 4
DEFAULT_FONT_FAMILY = "sans-serif"
DEFAULT_FONT_SERIF = [
    "Times New Roman",
    "Times",
    "Bitstream Vera Serif",
    "DejaVu Serif",
    "New Century Schoolbook",
    "Century Schoolbook L",
    "Utopia",
    "ITC Bookman",
    "Bookman",
    "Nimbus Roman No9 L",
    "Palatino",
    "Charter",
    "serif",
]
DEFAULT_FIGURE_FACE_COLOR = "white"  # figure facecolor; 0.75 is scalar gray
DEFAULT_LEGEND_FONT_SIZE = 23  # DEFAULT_FONT_SIZE
DEFAULT_AXES_LABEL_SIZE = DEFAULT_FONT_SIZE  # fontsize of the x any y labels
DEFAULT_TEXT_USE_TEX = False
LINE_ALPHA = 0.9
SAVE_FIGURES = False
FILE_EXTENSIONS = ["pdf", "png"]  # ,'eps']
FIGURES_DPI = 150
SHOW_FIGURES = False
FIGURE_PATH = "./plot/"

mpl.rcdefaults()
mpl.rcParams["lines.linewidth"] = DEFAULT_LINE_WIDTH
mpl.rcParams["lines.markersize"] = DEFAULT_MARKER_SIZE
mpl.rcParams["patch.linewidth"] = 1
mpl.rcParams["font.family"] = DEFAULT_FONT_FAMILY
mpl.rcParams["font.size"] = DEFAULT_FONT_SIZE
mpl.rcParams["font.serif"] = DEFAULT_FONT_SERIF
mpl.rcParams["text.usetex"] = DEFAULT_TEXT_USE_TEX
mpl.rcParams["axes.labelsize"] = DEFAULT_AXES_LABEL_SIZE
mpl.rcParams["axes.grid"] = True
mpl.rcParams["legend.fontsize"] = DEFAULT_LEGEND_FONT_SIZE
# opacity of of legend frame
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams["figure.facecolor"] = DEFAULT_FIGURE_FACE_COLOR
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
scale = 1.0
mpl.rcParams["figure.figsize"] = 23 * scale, 10 * scale  # 23, 18  # 12, 9
line_styles = 10 * ["b", "g", "r", "c", "y", "k", "m"]


def load_model():
    """Load the pinocchio model"""
    pinocchio_model_dir = join(
        dirname(dirname(dirname(dirname(dirname(str(abspath(__file__))))))), "models"
    )
    model_path = join(pinocchio_model_dir, "franka_description/robots")
    mesh_dir = pinocchio_model_dir
    urdf_filename = "franka2.urdf"
    urdf_model_path = join(join(model_path, "panda"), urdf_filename)

    robot = pin.RobotWrapper.BuildFromURDF(
        urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
    )
    rmodel1, [vmodel1, cmodel1] = robot.model, [
        robot.visual_model,
        robot.collision_model,
    ]
    q0 = pin.neutral(rmodel1)

    rmodel, [vmodel, cmodel] = pin.buildReducedModel(
        rmodel1, [vmodel1, cmodel1], [1, 9, 10], q0
    )

    ### CREATING THE SPHERE ON THE UNIVERSE
    OBSTACLE_RADIUS = 1.0e-1
    # OBSTACLE_POSE = pin.SE3.Identity()
    # OBSTACLE_POSE.translation = np.array([0.25, -0.425, 1.5])
    OBSTACLE_POSE = pin.SE3(pin.utils.rotate("x", np.pi), np.array([0, -0.2, 1.5]))
    OBSTACLE = hppfcl.Sphere(OBSTACLE_RADIUS)
    OBSTACLE_GEOM_OBJECT = pin.GeometryObject(
        "obstacle",
        rmodel.getFrameId("universe"),
        rmodel.frames[rmodel.getFrameId("universe")].parent,
        OBSTACLE,
        OBSTACLE_POSE,
    )
    ID_OBSTACLE = cmodel.addGeometryObject(OBSTACLE_GEOM_OBJECT)

    return rmodel, cmodel


class ResultAnalysisForAScene:
    def __init__(self, list_of_results) -> None:
        self._list_of_results = list_of_results

    def get_all_time_calc(self):
        self._time_calc_dict = {}
        for res in self._list_of_results:
            result_name = (
                str(res.get_Nnodes())
                + " nodes "
                + str(res.get_dt())
                + " dt "
                + str(res.get_max_iter())
                + " maxit "
            )
            self._time_calc_dict[result_name] = res.get_time_calc()

    def plot_time_calc(self):
        self.get_all_time_calc()
        for key, val in self._time_calc_dict.items():
            plt.plot(val, "o", label=key, linewidth=0.5)
        plt.legend()
        plt.xlabel("Time steps (ms)")
        plt.ylabel("Time taken for solving the OCP (s)")
        plt.title("Time taken for solving the OCP through time")
        plt.show()

    def get_all_min_distances(self, rmodel, cmodel):
        self._min_distance_dict = {}
        self._dict_of_collisions_objects = {}
        for res in self._list_of_results:
            self._min_distance_dict[
                res.get_name()
            ] = res.compute_minimal_distances_between_collision_pairs(rmodel, cmodel)
            for key in self._min_distance_dict[res.get_name()].keys():
                if key not in self._dict_of_collisions_objects.keys():
                    self._dict_of_collisions_objects[key] = [res.get_name()]
                else:
                    self._dict_of_collisions_objects[key].append(res.get_name())

    def plot_min_distances(self, rmodel, cmodel):
        subplot_dict = {
            1: None,
            2: [121, 122],
            3: [221, 222, 223],
            4: [221, 222, 223, 224],
            5: [321, 322, 323, 324, 325],
            6: [321, 322, 323, 324, 325, 326],
            7: [331, 332, 333, 334, 335, 336, 337],
            8: [331, 332, 333, 334, 335, 336, 337, 338],
            9: [331, 332, 333, 334, 335, 336, 337, 338, 339],
        }
        self.get_all_min_distances(rmodel, cmodel)
        subplot_number = subplot_dict[len(self._dict_of_collisions_objects)]
        for i, key in enumerate(self._dict_of_collisions_objects.keys()):
            plt.subplot(subplot_number[i])
            for kkey, value in self._min_distance_dict.items():
                if key in value:
                    plt.plot(value[key], label=kkey)
            plt.plot(np.zeros(len(value[key])), "--")
            plt.title(key)
        plt.legend()
        plt.xlabel("Time steps (ms)")
        plt.ylabel("Distance (m)")
        plt.suptitle("Distances minimal to the obstacle through time steps")
        plt.show()

    def plot_config(self):
        subplot = [331, 332, 333, 334, 335, 336, 337]
        Q1, Q2, Q3, Q4, Q5, Q6, Q7 = {}, {}, {}, {}, {}, {}, {}
        for res in self._list_of_results:
            Q1[res.get_name()] = [q[0] for q in res.get_Q()]
            Q2[res.get_name()] = [q[1] for q in res.get_Q()]
            Q3[res.get_name()] = [q[2] for q in res.get_Q()]
            Q4[res.get_name()] = [q[3] for q in res.get_Q()]
            Q5[res.get_name()] = [q[4] for q in res.get_Q()]
            Q6[res.get_name()] = [q[5] for q in res.get_Q()]
            Q7[res.get_name()] = [q[6] for q in res.get_Q()]

        for i, Q in enumerate([Q1, Q2, Q3, Q4, Q5, Q6, Q7]):
            plt.subplot(subplot[i])
            for key, value in Q.items():
                plt.plot(value, label=key)
            plt.title(key)
        plt.legend()
        plt.xlabel("Time steps (ms)")
        plt.ylabel("Configurations (rad)")
        plt.suptitle("Configurations through time steps")
        plt.show()

    def plot_velocities(self):
        subplot = [331, 332, 333, 334, 335, 336, 337]
        V1, V2, V3, V4, V5, V6, V7 = {}, {}, {}, {}, {}, {}, {}
        for res in self._list_of_results:
            V1[res.get_name()] = [V[0] for V in res.get_V()]
            V2[res.get_name()] = [V[1] for V in res.get_V()]
            V3[res.get_name()] = [V[2] for V in res.get_V()]
            V4[res.get_name()] = [V[3] for V in res.get_V()]
            V5[res.get_name()] = [V[4] for V in res.get_V()]
            V6[res.get_name()] = [V[5] for V in res.get_V()]
            V7[res.get_name()] = [V[6] for V in res.get_V()]

        for i, V in enumerate([V1, V2, V3, V4, V5, V6, V7]):
            plt.subplot(subplot[i])
            for key, value in V.items():
                plt.plot(value, label=key, linewidth=2.5)
                plt.title(key)

        plt.legend()
        plt.xlabel("Time steps (ms)")
        plt.ylabel("Velocities (rad/s)")
        plt.suptitle("Velocities through time steps")
        plt.show()

    def plot_controls(self):
        subplot = [331, 332, 333, 334, 335, 336, 337]
        U1, U2, U3, U4, U5, U6, U7 = {}, {}, {}, {}, {}, {}, {}
        for res in self._list_of_results:
            U1[res.get_name()] = [U[0] for U in res.get_U()]
            U2[res.get_name()] = [U[1] for U in res.get_U()]
            U3[res.get_name()] = [U[2] for U in res.get_U()]
            U4[res.get_name()] = [U[3] for U in res.get_U()]
            U5[res.get_name()] = [U[4] for U in res.get_U()]
            U6[res.get_name()] = [U[5] for U in res.get_U()]
            U7[res.get_name()] = [U[6] for U in res.get_U()]

        for i, U in enumerate([U1, U2, U3, U4, U5, U6, U7]):
            plt.subplot(subplot[i])
            for key, value in U.items():
                plt.plot(value, label=key)
                plt.title(key)
                plt.grid("on")

        plt.legend()
        plt.xlabel("Time steps (ms)")
        plt.ylabel("Torque (Nm)")
        plt.suptitle("Torque through time steps")
        plt.show()

    def plot_controls_custom(self):
        first = True
        subplot = [311, 312, 313]
        ylabel_titles = [
            "Link 5",
            "Link 6",
            "Link 7",
        ]
        U1, U2, U3, U4, U5, U6, U7 = {}, {}, {}, {}, {}, {}, {}
        for res in self._list_of_results:
            U1[res.get_name()] = [U[0] for U in res.get_U()]
            U2[res.get_name()] = [U[1] for U in res.get_U()]
            U3[res.get_name()] = [U[2] for U in res.get_U()]
            U4[res.get_name()] = [U[3] for U in res.get_U()]
            U5[res.get_name()] = [U[4] for U in res.get_U()]
            U6[res.get_name()] = [U[5] for U in res.get_U()]
            U7[res.get_name()] = [U[6] for U in res.get_U()]

        for i, U in enumerate([U5, U6, U7]):
            plt.subplot(subplot[i])
            for key, value in U.items():
                if "node" in key:
                    lab = "SQP"
                else:
                    lab = "FDDP with collision cost = " + key
                plt.plot(value, label=lab)
                plt.ylabel(ylabel_titles[i])
                plt.grid("on")
            if first:
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [2, 0, 1, 3]
                plt.legend(
                    [handles[idx] for idx in order],
                    [labels[idx] for idx in order],
                    loc=1,
                )
                first = False

        plt.xlabel("Time steps (ms)")
        plt.show()

    def plot_min_distances_custom(self, rmodel, cmodel):
        subplot_dict = {
            1: None,
            2: [121, 122],
            3: [311, 312, 313],
            4: [221, 222, 223, 224],
            5: [321, 322, 323, 324, 325],
            6: [321, 322, 323, 324, 325, 326],
            7: [331, 332, 333, 334, 335, 336, 337],
            8: [331, 332, 333, 334, 335, 336, 337, 338],
            9: [331, 332, 333, 334, 335, 336, 337, 338, 339],
        }
        xlabel_titles = [
            "Link 6",
            "Link 7",
            "Right finger",
        ]
        first = True
        self.get_all_min_distances(rmodel, cmodel)
        interesting_keys = [
            "panda2_link6_sc_2-obstacle",
            "panda2_link7_sc_1-obstacle",
            "panda2_rightfinger_0-obstacle",
        ]
        subplot_number = subplot_dict[len(interesting_keys)]
        for i, key in enumerate(interesting_keys):
            plt.subplot(subplot_number[i])
            plt.plot(np.zeros(4000), "--k", label="Collision limit", linewidth = 2.5)
            for kkey, value in self._min_distance_dict.items():
                if key in value:
                    if "csqp" in kkey:
                        lab = "CSQP"
                    elif "fddp" in kkey:
                        lab = "FDDP with collision cost = " + kkey.split("_")[0]
                    elif "sqp" in kkey:
                        lab = "SQP with collision cost = " + kkey.split("_")[0]
                    plt.plot(value[key], label=lab, linewidth=2.5)
                    plt.ylabel(xlabel_titles[i])

            plt.grid("on")
            if first:
                handles, labels = plt.gca().get_legend_handles_labels()
                order = [2, 1, 0, 4, 3]
                plt.legend(
                    [handles[idx] for idx in order],
                    [labels[idx] for idx in order],
                    bbox_to_anchor=(1.0, 1.5), loc="upper right",
           fancybox=True, shadow=True, borderaxespad=0  , ncol = 3)
                
                first = False
            # plt.axis("off")
        plt.xlabel("Time steps (ms)")
        # plt.suptitle("Distances minimal to the obstacle through time steps")
        plt.show()
        
        
    def plot_min_distances_custom1(self, rmodel, cmodel):
        xlabel_titles = [
            "Link 6",
            "Link 7",
            "Right finger",
        ]
        first = True
        self.get_all_min_distances(rmodel, cmodel)
        interesting_keys = [
            "panda2_link6_sc_2-obstacle",
            "panda2_link7_sc_1-obstacle",
            "panda2_rightfinger_0-obstacle",
        ]

        # LINK 6
        x = np.linspace(0,4,4000)
        fig, axs = plt.subplots(nrows=3,  sharex=True, figsize=(20,12))
        axs[0].plot(x,np.zeros(4000), "-.k", label="Collision limit", linewidth = 6.5)
        axs[1].plot(x,np.zeros(4000), "-.k", label="Collision limit", linewidth = 6.5)
        axs[2].plot(x,np.zeros(4000), "-.k", label="Collision limit", linewidth = 6.5)

        for kkey, value in self._min_distance_dict.items():
            if interesting_keys[0] in value:
                if "csqp" in kkey:
                    lab = "CSQP"
                elif "fddp" in kkey:
                    lab = "FDDP with collision cost = " + kkey.split("_")[0]
                elif "sqp" in kkey:
                    lab = "USQP with collision cost = " + kkey.split("_")[0]
                axs[0].plot(x,value[interesting_keys[0]][:-1], label=lab, linewidth=6.5)
                axs[0].set_ylabel(xlabel_titles[0])
                axs[0].grid("on")
                
        handles, labels = axs[0].get_legend_handles_labels()
        order = [2, 1, 5, 4, 3, 0]
        axs[0].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            bbox_to_anchor=(1.0, 1.5), loc="upper right",
    fancybox=True, shadow=True, borderaxespad=0  , ncol = 3)

        for kkey, value in self._min_distance_dict.items():
            if interesting_keys[1] in value:
                if "csqp" in kkey:
                    lab = "CSQP"
                elif "fddp" in kkey:
                    lab = "FDDP with collision cost = " + kkey.split("_")[0]
                elif "sqp" in kkey:
                    lab = "USQP with collision cost = " + kkey.split("_")[0]
                axs[1].plot(x,value[interesting_keys[1]][:-1], label=lab, linewidth=6.5)
                axs[1].set_ylabel(xlabel_titles[1])
                axs[1].grid("on")

        for kkey, value in self._min_distance_dict.items():
            if interesting_keys[2] in value:
                if "csqp" in kkey:
                    lab = "CSQP"
                elif "fddp" in kkey:
                    lab = "FDDP with collision cost = " + kkey.split("_")[0]
                elif "sqp" in kkey:
                    lab = "USQP with collision cost = " + kkey.split("_")[0]
                axs[2].plot(x,value[interesting_keys[2]][:-1], label=lab, linewidth=6.5)
                axs[2].set_ylabel(xlabel_titles[2])
                axs[2].grid("on")
                
        axs[1].set_xlim(1.5,2.5)
        axs[0].set_xlim(1.5,2.5)
        axs[2].set_xlim(1.5,2.5)
            # plt.axis("off")
        plt.xlabel("Time (s)")
        plt.savefig("distance_to_time.svg", bbox_inches="tight")
        plt.show()

    def plot_distances_to_targets_scene1(self, rmodel):
        x = np.linspace(0,4,4000)
        fig, axs = plt.subplots(nrows=2,  sharex=True, figsize=(10,6))
        for res in self._list_of_results:
            if "csqp" in res.get_name():
                    lab = "CSQP"
            elif "sqp" in res.get_name():
                    lab = "USQP with collision cost = " + res.get_name().split("_")[0]
            else:
                    lab = "FDDP with collision cost = " + res.get_name().split("_")[0]
            axs[0].plot(x,res.compute_distances_between_target1_scene1(rmodel)[:-1], label = lab, linewidth=6.5)
            axs[0].set_ylabel("Target 1")
            axs[0].grid("on")
        handles, labels = axs[0].get_legend_handles_labels()
        order = [1, 0, 4, 3, 2]
        axs[0].legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(1.0, 1.3), loc="upper right",
    fancybox=True, shadow=True, borderaxespad=0  , ncol = 3)
        for res in self._list_of_results:
            if "csqp" in res.get_name():
                    lab = "CSQP"
            elif "sqp" in res.get_name():
                    lab = "USQP with collision cost = " + res.get_name().split("_")[0]
            else:
                    lab = "FDDP with collision cost = " + res.get_name().split("_")[0]           
            axs[1].plot(x,res.compute_distances_between_target2_scene1(rmodel)[:-1], label = lab, linewidth=6.5)
        axs[1].set_ylabel("Target 2 ")
        axs[1].set_xlabel("Time (s)")
        
        plt.show()
if __name__ == "__main__":
    COMPARING_NODES = False
    SCENE1 = True
    SCENE2 = False
    COMPARING_CSQP_FDDP = False
    COMPARING_COST_FDDP = False
    COMPARING_SQP_FDDP_CSQP = True
    rmodel, cmodel = load_model()
    curr_path = dirname(str(abspath(__file__)))

    list_analysis = []
    if SCENE1:
        path = curr_path + "/scene1"
        if COMPARING_NODES:
            path += "/comparingnodes"
        elif COMPARING_CSQP_FDDP:
            path += "/comparingcsqpfddp"
        elif COMPARING_COST_FDDP:
            path += "/fddp_weight_col_comparing"
        elif COMPARING_SQP_FDDP_CSQP:
            path += "/comparingcsqpfddpsqp"
    if SCENE2:
        path = curr_path + "/scene2"
    for filename in listdir(path):
        f = join(path, filename)
        # checking if it is a file
        if isfile(f):
            list_analysis.append(Result(f))

    ana = ResultAnalysisForAScene(list_analysis)
    # ana.plot_time_calc()
    # ana.plot_min_distances(rmodel, cmodel)
    # ana.plot_config()
    # ana.plot_velocities()
    # ana.plot_controls_custom()
    ana.plot_min_distances_custom1(rmodel, cmodel)
    ana.plot_distances_to_targets_scene1(rmodel)