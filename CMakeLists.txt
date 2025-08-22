cmake_minimum_required(VERSION 3.8)
project(python_slam)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME} SCRIPTS_DESTINATION lib/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
  src/${PROJECT_NAME}/slam_node.py
  src/${PROJECT_NAME}/feature_extraction_node.py
  src/${PROJECT_NAME}/pose_estimation_node.py
  src/${PROJECT_NAME}/mapping_node.py
  src/${PROJECT_NAME}/localization_node.py
  src/${PROJECT_NAME}/loop_closure_node.py
  src/${PROJECT_NAME}/flight_integration_node.py
  src/${PROJECT_NAME}/basic_slam_pipeline.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install RViz config files
install(DIRECTORY rviz/
  DESTINATION share/${PROJECT_NAME}/rviz
)

# Install other resources
install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  find_package(ament_copyright REQUIRED)
  find_package(ament_flake8 REQUIRED)
  find_package(ament_pep257 REQUIRED)
  find_package(python3-pytest REQUIRED)

  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Add Python tests
  ament_add_pytest_test(${PROJECT_NAME}_tests tests/
    PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
    APPEND_ENV PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
  )
endif()

ament_package()
