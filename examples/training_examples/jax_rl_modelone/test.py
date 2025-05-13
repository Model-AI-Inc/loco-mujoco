import mujoco
import pathlib

try:
    # Construct the full path to your XML file
    # Assuming your script is in a directory, and 'modelone/assets/modelone.xml' is relative to that
    # Adjust this path as necessary
    xml_path = pathlib.Path(__file__).parent.parent.parent.parent / "loco_mujoco" / "models" / "modelone" / "modelone.xml" # Example path
    print(f"Attempting to load XML: {xml_path.resolve()}")
    
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    print("Vanilla MuJoCo loaded the model successfully.")

except Exception as e:
    print(f"Error loading model with vanilla MuJoCo: {e}")
