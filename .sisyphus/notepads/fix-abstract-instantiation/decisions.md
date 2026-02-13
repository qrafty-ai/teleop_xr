# Decisions - Fix Abstract Instantiation

## Joint Count in MockRobot

Used 10 joints in 'actuated_joint_names' to match the output of
'get_default_config' which returns 'jnp.zeros(10)'.
