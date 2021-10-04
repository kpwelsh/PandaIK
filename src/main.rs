use std::time::{Instant};


mod robot;
use robot::state::{State, NamedTransform};
use robot::constraints::{ParsedConstraints};

extern crate nalgebra as na;
use na::{Vector3};

use serde_yaml;
use std::fs::File;
use std::io::BufReader;

fn initial_state(u: &[f64; 7]) -> State {
    let zeros = [Some(0.0); 7];

    State { 
        joint_position: u.map(|ui| {Some(ui)}), 
        joint_velocity: zeros, 
        joint_acceleration: zeros, 
        transforms: None
    }
}

fn update_state(state: &mut State, u: &[f64; 7], dt : f64) {
    let mut vel = [0.0; 7];
    let mut accel = [0.0; 7];
    for i in 0..7 {
        vel[i] = (u[i] - state.joint_position[i].unwrap()) / dt;
        accel[i] = (vel[i] - state.joint_velocity[i].unwrap()) / dt;
    }

    state.joint_position = u.map(|ui| {Some(ui)});
    state.joint_velocity = vel.map(|vi| {Some(vi)});
    state.joint_acceleration = accel.map(|ai| {Some(ai)});
}


fn main() {
    let file = File::open("constraints.yaml").unwrap();
    let rdr = BufReader::new(file);
    let constraints : robot::constraints::RobotConstraints = serde_yaml::from_reader(rdr).unwrap();

    let chain = k::Chain::<f64>::from_urdf_file("panda.urdf").unwrap();
    
    // Create a set of joints from end joint
    let ee = chain.find("panda_joint8").unwrap();
    let arm = k::SerialChain::from_end(ee);
    println!("chain: {}", arm);


    let mut controller = robot::ik::Controller::new(arm, Some(ParsedConstraints::from_robot_constraints(constraints)));


    let goal = NamedTransform {
        name: "panda_joint8".to_string(),
        position: Some(Vector3::from([-10., 0.0, 0.4])),
        rotation: None
    };

    let mut state = initial_state(&[0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]);

    let start = Instant::now();
    let n = 10000;
    let dt = 1e-3;
    println!("{:?}", state.joint_position);
    for _i in 0..n {

        let mut desired_state = State::new();
        desired_state.transforms = Some(vec![goal.clone()]);

        let u = controller.update(&state, &desired_state, dt);
        update_state(&mut state, &u, dt);
    }
    println!("{:?}", state.joint_position);
    let duration = start.elapsed();
    println!("{}", 1e6 * n as f64 / (duration.as_micros() as f64));
}
