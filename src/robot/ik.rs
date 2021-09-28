use std::f64::{INFINITY, NEG_INFINITY};

use hyperdual::Const;
use k::{self, Constraints, UnitQuaternion};
use optimization_engine::constraints::{Constraint, Rectangle};
use optimization_engine::{constraints::NoConstraints, panoc::*, *};
extern crate nalgebra as na;
use na::{SimdPartialOrd, Vector, Vector3};
use crate::robot::state::{State};
use crate::robot::constraints::RobotConstraints;
use super::constraints::{JointConstraint, ParsedConstraints};



type CostFunctionType = dyn Fn(&[f64], &mut f64) -> Result<(), SolverError>;
type Arm = k::SerialChain<f64>;


fn finite_difference(f: &dyn Fn(&[f64], &mut f64) -> Result<(), SolverError>, u: &[f64], grad: &mut [f64]) -> Result<(), SolverError> {
    let h = 1e-6;
    let mut f0 = 0.0;
    f(u, &mut f0).unwrap();

    let mut x = [0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    for i in 0..7 {
        x[i] = u[i];
    }

    for i in 0..7 {
        let mut fi = 0.0;
        x[i] += h;
        f(&x, &mut fi).unwrap();
        grad[i] = (fi - f0) / h;
        x[i] -= h;
    }

    Ok(())
}


fn joint_constraint_cost(u: &[f64], jcs: &[JointConstraint; 7]) -> f64 {
    let mut c = 0.0;
    for (&v, jc) in u.iter().zip(jcs) {
        let center = jc.min.unwrap() + (jc.max.unwrap() - jc.min.unwrap()) / 2.;
        let w = jc.max.unwrap() - jc.min.unwrap();
        c += (2. * (v - center) / w).powf(50.);
    }
    c
}


fn finite_diff(lu: &[Option<f64>; 7], u: &[f64], dt: f64) -> [f64; 7] {
    let mut vel = [0.0; 7];

    for ((v, &olu), x) in vel.iter_mut().zip(lu).zip(u) {
        *v = (x - olu.unwrap()) / dt;
    }

    vel
}

fn constraint_cost(constriants : &Option<RobotConstraints>, state: &State, u: &[f64], dt: f64) -> f64 {
    let mut c = 0.0;

    if let Some(constraints) = &constriants {
        let vel = finite_diff(&state.joint_position, u, dt);
        let accel = finite_diff(&state.joint_velocity, &vel, dt);
        let jerk = finite_diff(&state.joint_acceleration, &accel, dt);

        // Position Constraints
        if let Some(jcs) = &constraints.joint_value {
            c += joint_constraint_cost(u, &jcs);
        }

        // Velocity Constraints
        if let Some(jcs) = &constraints.joint_vel {
            c += joint_constraint_cost(&vel, &jcs);
        }

        // Acceleration Constraints
        if let Some(jcs) = &constraints.joint_accel {
            c += joint_constraint_cost(&accel, &jcs);
        }

        // Jerk Constraints
        if let Some(jcs) = &constraints.joint_jerk {
            c += joint_constraint_cost(&jerk, &jcs);
        }
    }
    c
}


fn position_cost(current_position: &Vector3<f64>, desired_position: &Vector3<f64>) -> f64 {
    (current_position - desired_position).norm_squared()
}

fn rotation_cost(current_rotation: &UnitQuaternion<f64>, desired_rotation: &UnitQuaternion<f64>) -> f64 {
    current_rotation.angle_to(desired_rotation).powf(2.0)
}

fn goal_cost(desired_state: &State, arm: &Arm, dt: f64) -> f64 {
    let mut c = 0.0;
    if let Some(name_transforms) = &desired_state.transforms {
        for named_trans in name_transforms.iter() {
            let trans = arm.find(&named_trans.name).unwrap().world_transform().unwrap();

            if let Some(desired_position) = named_trans.position {
                c += position_cost(&trans.translation.vector, &desired_position);
            }
            
            if let Some(desired_rotation) = named_trans.rotation {
                c += rotation_cost(&trans.rotation, &desired_rotation);
            }
        }
    }
    c
}

fn compute_bounds(state: &State, constraints: &ParsedConstraints, dt: f64) -> ([f64;7], [f64;7]) {
    let q = Vector::from(state.joint_position.map(|p| p.unwrap()));
    let dq = Vector::from(state.joint_velocity.map(|p| p.unwrap()));
    let ddq = Vector::from(state.joint_acceleration.map(|p| p.unwrap()));

    let dt2 = dt * dt;
    let dt3 = dt2 * dt;

    let forward_stopping_distance = constraints.joint_jerk_min * dt3 +
     ddq*dt2 + dq*dt;
    let reverse_stopping_distance = constraints.joint_jerk_max * dt3 + ddq*dt2 + dq*dt;

    let mut lower_bounds = Vec::new();
    lower_bounds.push(constraints.joint_min);
    lower_bounds.push(constraints.joint_vel_min*dt + q);
    lower_bounds.push(constraints.joint_accel_min*dt2 + dq*dt + q);
    lower_bounds.push(constraints.joint_jerk_min*dt3 + ddq*dt2 + dq*dt + q);
    let mut upper_bounds = Vec::new();
    upper_bounds.push(constraints.joint_max);
    upper_bounds.push(constraints.joint_vel_max*dt + q);
    upper_bounds.push(constraints.joint_accel_max*dt2 + dq*dt + q);
    upper_bounds.push(constraints.joint_jerk_max*dt3 + ddq*dt2 + dq*dt + q);

    let mut lower_bound = [NEG_INFINITY; 7];
    let mut upper_bound = [INFINITY; 7];

    for i in 0..4 {
        for j in 0..7 {
            lower_bound[j] = lower_bound[j].max(lower_bounds[i][j]);
            upper_bound[j] = upper_bound[j].min(upper_bounds[i][j]);
        }
    }
    println!("q: {:?}", q);
    println!("dq: {:?}", dq);
    println!("ddq: {:?}", ddq);
    println!("Bounds: {:?}", Vector::from(upper_bound) - Vector::from(lower_bound));

    (lower_bound, upper_bound)
}


pub struct Controller {
    panoc_cache : PANOCCache,
    arm : Arm,
    last_state : Option<State>,
    constriants : Option<ParsedConstraints>
}

impl Controller {
    pub fn new(arm: Arm, constraints : Option<ParsedConstraints>) -> Self {
        Controller {
            panoc_cache : panoc::PANOCCache::new(7, 1e-4, 100),
            arm: arm,
            last_state: None,
            constriants: constraints
        }
    }
    
    pub fn set_state(&mut self, state: State) {
        self.last_state = Some(state);
    }

    pub fn update(&mut self, state: &State, desired_state: &State, dt: f64) -> [f64; 7] {

        let arm = &mut self.arm;

        let cost = |u: &[f64], c: &mut f64| {
            if let Err(err) = arm.set_joint_positions(u) {
                println!("{}|{:?}", err, u);
                panic!();
            }
            arm.update_transforms();
            *c += goal_cost(desired_state, &arm, dt);
            Ok(())
        };

        let dcost = |u: &[f64], grad: &mut [f64]| {
            finite_difference(&cost, u, grad)
        };

        
        let mut u = [0.0; 7];
        for (ui, qi) in u.iter_mut().zip(state.joint_position) {
            *ui = qi.unwrap();
        }
        
        let status;

        // Got kinda fed up with the types of this stuff and just copied some code.
        // Probably should use a default Rectangle with +-Inf bounds instead of "NoConstraints"
        if let Some(constraints) = &self.constriants {
            let (lb, ub) = compute_bounds(state, constraints, dt);
            let bounds = Rectangle::new(Some(&lb), Some(&ub));
            let problem = Problem::new(
                &bounds,
                 dcost, 
                cost
            );
            let mut panoc = PANOCOptimizer::new(problem, &mut self.panoc_cache)
                                .with_max_iter(10);
            status = panoc.solve(&mut u).unwrap();
        } else {
            let bounds = NoConstraints::new();
            let problem = Problem::new(
                &bounds,
                 dcost, 
                cost
            );
            let mut panoc = PANOCOptimizer::new(problem, &mut self.panoc_cache)
                                .with_max_iter(10);
            status = panoc.solve(&mut u).unwrap();
        }
        u
    }
}