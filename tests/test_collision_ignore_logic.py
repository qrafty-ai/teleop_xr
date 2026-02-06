import pyroki as pk
import yourdfpy
import io


def create_mock_urdf():
    urdf_str = """
    <robot name="test">
      <link name="link1">
        <collision>
          <origin xyz="0 0 0"/>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
      <link name="link2">
        <collision>
          <origin xyz="1 0 0"/>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
      <joint name="joint1" type="fixed">
        <parent link="link1"/>
        <child link="link2"/>
      </joint>
      <link name="link3">
        <collision>
          <origin xyz="2 0 0"/>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
      <joint name="joint2" type="fixed">
        <parent link="link2"/>
        <child link="link3"/>
      </joint>
      <link name="link4">
        <collision>
          <origin xyz="3 0 0"/>
          <geometry><sphere radius="0.1"/></geometry>
        </collision>
      </link>
      <joint name="joint3" type="fixed">
        <parent link="link3"/>
        <child link="link4"/>
      </joint>
    </robot>
    """
    return yourdfpy.URDF.load(io.StringIO(urdf_str))


def get_active_pairs(coll):
    """
    Helper to extract active collision pairs as link name tuples.
    """
    active_pairs = []
    for i, j in zip(coll.active_idx_i, coll.active_idx_j):
        # Map sphere/geom index to link name
        # coll._geom_to_link_idx maps geometry index to link index
        # coll.link_names maps link index to name
        name_i = coll.link_names[coll._geom_to_link_idx[i]]
        name_j = coll.link_names[coll._geom_to_link_idx[j]]
        active_pairs.append(tuple(sorted((name_i, name_j))))
    return set(active_pairs)


def test_ignore_immediate_adjacents():
    urdf = create_mock_urdf()
    sphere_decomp = {
        "link1": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.1]},
        "link2": {"centers": [[1.0, 0.0, 0.0]], "radii": [0.1]},
        "link3": {"centers": [[2.0, 0.0, 0.0]], "radii": [0.1]},
        "link4": {"centers": [[3.0, 0.0, 0.0]], "radii": [0.1]},
    }

    # Case 1: ignore_immediate_adjacents=True
    coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomp, urdf, ignore_immediate_adjacents=True
    )
    active_pairs = get_active_pairs(coll)

    # Adjacent pairs (should be ignored):
    # (link1, link2), (link2, link3), (link3, link4)
    assert ("link1", "link2") not in active_pairs
    assert ("link2", "link3") not in active_pairs
    assert ("link3", "link4") not in active_pairs

    # Non-adjacent pairs (should be active):
    # (link1, link3), (link1, link4), (link2, link4)
    assert ("link1", "link3") in active_pairs
    assert ("link1", "link4") in active_pairs
    assert ("link2", "link4") in active_pairs


def test_user_ignore_pairs():
    urdf = create_mock_urdf()
    sphere_decomp = {
        "link1": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.1]},
        "link2": {"centers": [[1.0, 0.0, 0.0]], "radii": [0.1]},
        "link3": {"centers": [[2.0, 0.0, 0.0]], "radii": [0.1]},
        "link4": {"centers": [[3.0, 0.0, 0.0]], "radii": [0.1]},
    }

    # Case 2: user_ignore_pairs=[("link1", "link3")]
    # Note: ignore_immediate_adjacents=False to strictly test user_ignore_pairs
    coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomp,
        urdf,
        ignore_immediate_adjacents=False,
        user_ignore_pairs=(("link1", "link3"),),
    )
    active_pairs = get_active_pairs(coll)

    # Specified pair (should be ignored)
    assert ("link1", "link3") not in active_pairs

    # Adjacent pairs (should be active since ignore_immediate_adjacents=False)
    assert ("link1", "link2") in active_pairs
    assert ("link2", "link3") in active_pairs
    assert ("link3", "link4") in active_pairs


def test_combined_ignore():
    urdf = create_mock_urdf()
    sphere_decomp = {
        "link1": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.1]},
        "link2": {"centers": [[1.0, 0.0, 0.0]], "radii": [0.1]},
        "link3": {"centers": [[2.0, 0.0, 0.0]], "radii": [0.1]},
        "link4": {"centers": [[3.0, 0.0, 0.0]], "radii": [0.1]},
    }

    # Case 3: Combined
    coll = pk.collision.RobotCollision.from_sphere_decomposition(
        sphere_decomp,
        urdf,
        ignore_immediate_adjacents=True,
        user_ignore_pairs=(("link1", "link4"),),
    )
    active_pairs = get_active_pairs(coll)

    # Adjacent (ignored)
    assert ("link1", "link2") not in active_pairs
    assert ("link2", "link3") not in active_pairs
    assert ("link3", "link4") not in active_pairs

    # User specified (ignored)
    assert ("link1", "link4") not in active_pairs

    # Remaining (active)
    assert ("link1", "link3") in active_pairs
    assert ("link2", "link4") in active_pairs
