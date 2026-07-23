#include <iostream>
#include <functional>
#include <memory>
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <condition_variable>
#include <mutex>

#include "hiwin_interfaces/srv/robot_command.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp/executors/multi_threaded_executor.hpp"
// #include "rclcpp_action/rclcpp_action.hpp"
// #include "rclcpp_components/register_node_macro.hpp"
#include "hiwin_libmodbus/hiwin_libmodbus.hpp"

#include "hiwin_libmodbus//visibility_control.h"

HiwinLibmodbus hiwinlibmodbus;
class HiwinlibmodbusServiceServer : public rclcpp::Node
{
    
    public:
    HiwinlibmodbusServiceServer() : Node("hiwinmodbus_server")
    {
        server_ = create_service<hiwin_interfaces::srv::RobotCommand>(
                "hiwinmodbus_service", std::bind(&HiwinlibmodbusServiceServer::hiwinmodbus_execute, this,
                                        std::placeholders::_1, std::placeholders::_2));
        stop_callback_group_ = create_callback_group(
            rclcpp::CallbackGroupType::MutuallyExclusive);
        stop_server_ = create_service<hiwin_interfaces::srv::RobotCommand>(
                "hiwinmodbus_stop_service",
                std::bind(&HiwinlibmodbusServiceServer::hiwinmodbus_stop, this,
                          std::placeholders::_1, std::placeholders::_2),
                rmw_qos_profile_services_default, stop_callback_group_);
        rearm_server_ = create_service<hiwin_interfaces::srv::RobotCommand>(
                "hiwinmodbus_rearm_service",
                std::bind(&HiwinlibmodbusServiceServer::hiwinmodbus_rearm, this,
                          std::placeholders::_1, std::placeholders::_2),
                rmw_qos_profile_services_default, stop_callback_group_);
        output_timer_thread_ = std::thread(
            &HiwinlibmodbusServiceServer::output_timer_loop, this);

    }

    ~HiwinlibmodbusServiceServer() override
    {
        {
            std::unique_lock<std::mutex> lock(mutex);
            int stopped_arm_state = 0;
            if (!stop_motion_and_clear_outputs(stopped_arm_state)) {
                RCLCPP_ERROR(
                    get_logger(),
                    "Motion/output shutdown verification failed");
            }
        }
        {
            std::lock_guard<std::mutex> timer_lock(timer_mutex_);
            timers_shutting_down_ = true;
        }
        timer_condition_.notify_all();
        if (output_timer_thread_.joinable()) {
            output_timer_thread_.join();
        }
    }

    private:
                
        int digital_state;
        std::vector<double> current_pos;
        std::vector<double> command;
        int command_type;
        std::mutex mutex;
        const std::vector<int> controlled_outputs_{301, 300, 303, 304, 305};
        struct TimedOutput {
            int output;
            std::chrono::steady_clock::time_point deadline;
        };
        std::vector<TimedOutput> scheduled_outputs_;
        std::thread output_timer_thread_;
        std::mutex timer_mutex_;
        std::condition_variable timer_condition_;
        bool timers_shutting_down_{false};
        std::atomic<uint64_t> stop_generation_{0};
        std::atomic<bool> stop_latched_{false};
        // std::array<double, 6> command;
        // double command[6];
        bool turn_output_off_with_retries(int d_o) {
            for (int attempt = 0; attempt < 3; ++attempt) {
                if (hiwinlibmodbus.DO(d_o, 0)) {
                    return true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            return false;
        }

        bool stop_motion_and_clear_outputs(int & stopped_arm_state) {
            bool stop_sent = false;
            for (int attempt = 0; attempt < 3 && !stop_sent; ++attempt) {
                stop_sent = hiwinlibmodbus.Motion_Stop();
                if (!stop_sent) {
                    std::this_thread::sleep_for(
                        std::chrono::milliseconds(50));
                }
            }
            bool outputs_off = true;
            for (const int output : controlled_outputs_) {
                outputs_off = turn_output_off_with_retries(output)
                    && outputs_off;
            }
            const auto stop_deadline = std::chrono::steady_clock::now()
                + std::chrono::seconds(3);
            do {
                hiwinlibmodbus.Arm_State_REGISTERS(stopped_arm_state);
                if (stopped_arm_state == 1 || stopped_arm_state == 3) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            } while (std::chrono::steady_clock::now() < stop_deadline);
            const bool motion_stopped =
                stopped_arm_state == 1 || stopped_arm_state == 3;
            return stop_sent && motion_stopped && outputs_off;
        }

        bool schedule_output_off(int d_o, int time) {
            std::lock_guard<std::mutex> timer_lock(timer_mutex_);
            if (timers_shutting_down_) {
                return false;
            }
            scheduled_outputs_.push_back({
                d_o,
                std::chrono::steady_clock::now()
                    + std::chrono::milliseconds(time * 100),
            });
            timer_condition_.notify_one();
            return true;
        }

        void output_timer_loop() {
            std::unique_lock<std::mutex> timer_lock(timer_mutex_);
            while (!timers_shutting_down_) {
                if (scheduled_outputs_.empty()) {
                    timer_condition_.wait(timer_lock);
                    continue;
                }
                const auto earliest = std::min_element(
                    scheduled_outputs_.begin(), scheduled_outputs_.end(),
                    [](const TimedOutput & lhs, const TimedOutput & rhs) {
                        return lhs.deadline < rhs.deadline;
                    });
                const auto earliest_deadline = earliest->deadline;
                if (timer_condition_.wait_until(
                        timer_lock, earliest_deadline)
                        != std::cv_status::timeout) {
                    continue;
                }

                const auto now = std::chrono::steady_clock::now();
                std::vector<int> due_outputs;
                auto output = scheduled_outputs_.begin();
                while (output != scheduled_outputs_.end()) {
                    if (output->deadline <= now) {
                        due_outputs.push_back(output->output);
                        output = scheduled_outputs_.erase(output);
                    } else {
                        ++output;
                    }
                }
                timer_lock.unlock();
                for (const int due_output : due_outputs) {
                    std::unique_lock<std::mutex> output_lock(mutex);
                    if (!turn_output_off_with_retries(due_output)) {
                        RCLCPP_ERROR(
                            get_logger(),
                            "Timed shutdown failed for DO %d", due_output);
                    }
                }
                timer_lock.lock();
            }
        }

        rclcpp::Service<hiwin_interfaces::srv::RobotCommand>::SharedPtr server_;
        rclcpp::Service<hiwin_interfaces::srv::RobotCommand>::SharedPtr stop_server_;
        rclcpp::Service<hiwin_interfaces::srv::RobotCommand>::SharedPtr rearm_server_;
        rclcpp::CallbackGroup::SharedPtr stop_callback_group_;

        void hiwinmodbus_stop(
                  const std::shared_ptr<hiwin_interfaces::srv::RobotCommand::Request>,
                  std::shared_ptr<hiwin_interfaces::srv::RobotCommand::Response> response)
        {
            // Latch before waiting for the Modbus mutex.  A motion request
            // already queued behind this stop must not start afterwards.
            stop_latched_.store(true);
            std::unique_lock<std::mutex> lock(mutex);
            int stopped_arm_state = 0;
            const bool stop_confirmed =
                stop_motion_and_clear_outputs(stopped_arm_state);
            if (stop_confirmed) {
                stop_generation_.fetch_add(1);
            }
            response->arm_state = stopped_arm_state;
            response->digital_state = stop_confirmed ? 1 : 255;
            if (response->digital_state == 1) {
                RCLCPP_WARN(get_logger(), "Motion stop confirmed");
            } else {
                RCLCPP_ERROR(
                    get_logger(),
                    "Motion stop or output-off verification failed");
            }
        }

        void hiwinmodbus_rearm(
                  const std::shared_ptr<hiwin_interfaces::srv::RobotCommand::Request>,
                  std::shared_ptr<hiwin_interfaces::srv::RobotCommand::Response> response)
        {
            std::unique_lock<std::mutex> lock(mutex);
            int current_arm_state = 0;
            hiwinlibmodbus.Arm_State_REGISTERS(current_arm_state);
            bool outputs_off = current_arm_state == 1;
            if (outputs_off) {
                for (const int output : controlled_outputs_) {
                    outputs_off = turn_output_off_with_retries(output)
                        && outputs_off;
                }
            }
            if (current_arm_state == 1 && outputs_off) {
                stop_latched_.store(false);
                response->digital_state = 1;
                RCLCPP_INFO(get_logger(), "Motion command latch re-armed");
            } else {
                response->digital_state = 255;
                RCLCPP_ERROR(
                    get_logger(),
                    "Motion re-arm rejected: arm is not IDLE or outputs are not OFF");
            }
            response->arm_state = current_arm_state;
        }

        void hiwinmodbus_execute(const std::shared_ptr<hiwin_interfaces::srv::RobotCommand::Request> request,    
                  std::shared_ptr<hiwin_interfaces::srv::RobotCommand::Response>     response)  
        {
            std::unique_lock<std::mutex> lock(mutex);
            const uint64_t command_stop_generation = stop_generation_.load();
            const bool is_motion_command =
                request->cmd_mode == 2 || request->cmd_mode == 3 ||
                request->cmd_mode == 4 || request->cmd_mode == 6 ||
                request->cmd_mode == 7;
            const bool is_output_on = request->cmd_mode == 5
                && request->digital_output_cmd != 0;
            if ((is_motion_command || is_output_on)
                    && stop_latched_.load()) {
                response->arm_state = 3;
                response->digital_state = 255;
                RCLCPP_ERROR(
                    get_logger(),
                    "Motion/output command rejected while stop latch is active");
                return;
            }
            // std::cout<<request->holding<<std::endl;
            if (request->cmd_mode == 1){
                hiwinlibmodbus.MOTOR_EXCITE();
            }
            else if (request->cmd_mode == 2){
                if(request->cmd_type==0){
                    command = {request->joints[0], request->joints[1], request->joints[2],
                    request->joints[3], request->joints[4], request->joints[5]};
                }
                else if(request->cmd_type==1){
                    command={request->pose.linear.x, request->pose.linear.y, request->pose.linear.z,
                    request->pose.angular.x, request->pose.angular.y, request->pose.angular.z};
                }

                if (!hiwinlibmodbus.PTP(
                        request->cmd_type, request->velocity,
                        request->acceleration, request->tool, request->base,
                        command)) {
                    response->arm_state = 3;
                    response->digital_state = 255;
                    RCLCPP_ERROR(get_logger(), "PTP command write failed");
                    return;
                }
            }
            else if (request->cmd_mode == 3){
                if(request->cmd_type==0){
                    command = {request->joints[0], request->joints[1], request->joints[2],
                    request->joints[3], request->joints[4], request->joints[5]};
                }
                else if(request->cmd_type==1){
                    command={request->pose.linear.x, request->pose.linear.y, request->pose.linear.z,
                    request->pose.angular.x, request->pose.angular.y, request->pose.angular.z};
                }
                if (!hiwinlibmodbus.LIN(
                        request->cmd_type, request->velocity,
                        request->acceleration, request->tool, request->base,
                        command)) {
                    response->arm_state = 3;
                    response->digital_state = 255;
                    RCLCPP_ERROR(get_logger(), "LIN command write failed");
                    return;
                }
            }
            else if (request->cmd_mode == 4){
                hiwinlibmodbus.CIRC(request->velocity, request->acceleration, request->tool, request->base, request->circ_s, request->circ_end); 
            }
            /************* Discret_e Input *************/
              /*********************************
                    S0
              value:0 ~ 255 -> S0[1] ~ [256]
                0 or 1    -> R 
              ----------------------------------
                    DI
              value:300 ~ 555 -> DI[1] ~ [256]
                0 or 1      -> R
              **********************************/
         
              /************* Coil *************/
              /*********************************
                    DI
              value:0 ~ 255  -> DI[1] ~ [256]
                0 or 65280 -> R/W 
              ----------------------------------
                    DO
              value:300 ~ 555 -> DO[1] ~ [256]
                0 or 65280  -> R/W 
              **********************************/
            else if (request->cmd_mode == 5){
                const int d_o = request->digital_output_pin+299;
                bool output_allowed = true;
                if (request->digital_output_cmd != 0) {
                    // Serialize output activation with robot motion and only
                    // energize an output while the controller reports IDLE.
                    int output_arm_state = 0;
                    hiwinlibmodbus.Arm_State_REGISTERS(output_arm_state);
                    output_allowed = output_arm_state == 1;
                }
                const bool output_verified = output_allowed &&
                    hiwinlibmodbus.DO(d_o, request->digital_output_cmd);
                response->digital_state = output_verified ?
                    request->digital_output_cmd : 255;
                if (output_allowed && request->digital_output_cmd != 0 &&
                    request->do_timer != 0) {
                    try {
                        if (!schedule_output_off(d_o, request->do_timer)) {
                            response->digital_state = 255;
                            turn_output_off_with_retries(d_o);
                        }
                    } catch (const std::exception & error) {
                        response->digital_state = 255;
                        turn_output_off_with_retries(d_o);
                        RCLCPP_ERROR(
                            get_logger(), "Unable to schedule DO %d cutoff: %s",
                            d_o, error.what());
                    }
                }
                if (output_allowed && request->digital_output_cmd != 0 &&
                    !output_verified) {
                    // The coil write may have succeeded before trigger or
                    // readback failed.  Attempt OFF immediately; the timer
                    // above remains a second software cutoff.
                    turn_output_off_with_retries(d_o);
                }
                request->holding = false;
            }
        /*********************
          value   joint 
          0~5 -> A1~A6 
          value  Cartesian
          6~11 -> XYZABC 
        *********************/
            else if (request->cmd_mode == 6){
                hiwinlibmodbus.HOME();
            }
            else if (request->cmd_mode == 7){
                hiwinlibmodbus.JOG(request->jog_joint, request->jog_dir); 
            }
            else if (request->cmd_mode == 8){
                hiwinlibmodbus.getArmJoints(current_pos); 
                response->current_position = current_pos;
            }
            else if (request->cmd_mode == 9){
                hiwinlibmodbus.getArmPose(current_pos); 
                std::cout<<"----------------------------------"<<std::endl;
                response->current_position = current_pos;
            }
            else if (request->cmd_mode == 10){
                request->holding = false;
                rclcpp::shutdown();
            }
            else if (request->cmd_mode == 11){
                request->holding = true;
            }
            else if (request->cmd_mode == 12){
                hiwinlibmodbus.Read_DI(299+request->digital_input_pin, digital_state);
                response->digital_state = digital_state;
                request->holding = false;
            }
            else if (request->cmd_mode == 13){
                command={request->pose.linear.x, request->pose.linear.y, request->pose.linear.z,
                request->pose.angular.x, request->pose.angular.y, request->pose.angular.z};
                hiwinlibmodbus.SET_BASE(request->base_num, command);
                request->holding = false;
            }
            else if (request->cmd_mode == 14){
                command={request->pose.linear.x, request->pose.linear.y, request->pose.linear.z,
                request->pose.angular.x, request->pose.angular.y, request->pose.angular.z};
                hiwinlibmodbus.SET_TOOL(request->tool_num, command);
                request->holding = false;
            }
            else if (request->cmd_mode == 15){
                stop_latched_.store(true);
                int stopped_arm_state = 0;
                const bool stop_confirmed =
                    stop_motion_and_clear_outputs(stopped_arm_state);
                if (stop_confirmed) {
                    stop_generation_.fetch_add(1);
                }
                response->arm_state = stopped_arm_state;
                response->digital_state = stop_confirmed ? 1 : 255;
                request->holding = false;
            }
            if (request->holding == true){
                // Do not hold the Modbus mutex for the entire move.  The
                // separate stop service and output-off timers must be able to
                // preempt the polling loop between short register reads.
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                const auto motion_deadline = std::chrono::steady_clock::now()
                    + std::chrono::seconds(120);
                while(rclcpp::ok()){
                    int polled_arm_state = 0;
                    {
                        std::unique_lock<std::mutex> poll_lock(mutex);
                        hiwinlibmodbus.Arm_State_REGISTERS(polled_arm_state); // return arm_state
                    }
                    // int arm_state = hiwinlibmodbus.Check_Arm_State();
                    response->arm_state = polled_arm_state;
                    if (polled_arm_state == 1 ||
                        stop_latched_.load() ||
                        stop_generation_.load() != command_stop_generation ||
                        std::chrono::steady_clock::now() >= motion_deadline) {
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            else{
                int current_arm_state = 0;
                hiwinlibmodbus.Arm_State_REGISTERS(current_arm_state); // return arm_state{
                response->arm_state = current_arm_state;
                }
          RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "sending back response");
        }
};



int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  if (!hiwinlibmodbus.libModbus_Connect("192.168.0.1")) {
    RCLCPP_ERROR(rclcpp::get_logger("rclcpp"),
                 "Unable to connect to the Hiwin controller");
    return 1;
  }
  hiwinlibmodbus.Holding_Registers_init();

  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Ready to recieve commands.");

  auto server = std::make_shared<HiwinlibmodbusServiceServer>();
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 2);
  executor.add_node(server);
  executor.spin();
  executor.remove_node(server);
  server.reset();
  hiwinlibmodbus.Modbus_Close();
}
