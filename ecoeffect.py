import datetime as dt

# import folium
import heat.landsatxplore
import json
from heat.landsatxplore.api import API
from heat.landsatxplore.earthexplorer import EarthExplorer
import os
from leafmap import builtin_legends
import leafmap.leafmap as leafmap
import leafmap.foliumap as foliumap
from ipyleaflet import Map, DrawControl
from shapely.geometry import shape, Polygon, Point, box
import panel as pn
from panel.theme import Bootstrap, Material, Native

# import ipyleaflet
import json

# from ipyleaflet import Map, GeoJSON, GeoData, basemaps, LayersControl
import geopandas as gpd
import os
from hydro.utils import Utils
import asyncio
import glob
import traceback
import uuid


# define project structure
PROJECT_DIR_STRUCTURE = {
    "Urban greening": [
        "input_data",
        "output_data",
        "input_data/tasmax",
        "input_data/tasmin",
        "input_data/srad",
    ],
    "default": [
        "clim",
        "soil",
        "elevation",
        "models",
        "output_data",
        "land_cover",
        "scratch",
        "shapes",
    ],
}


# ====================================================================================================================================================================
# Initialize the widgets and UI components
def create_buttons():
    return {
        "download": pn.widgets.Button(
            name="Download data for scenario analysis",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
        ),
        "preprocess": pn.widgets.Button(
            name="Preprocess data for scenario analysis",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
        ),
        "train": pn.widgets.Button(
            name="Train_model",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
        ),
        "evaluate": pn.widgets.Button(
            name="Show model performance results",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
            sizing_mode="stretch_width",
        ),
        "visualize": pn.widgets.Button(
            name="Visualize downloaded data",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
        ),
        "simulate": pn.widgets.Button(
            name="Run scenario",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
            sizing_mode="stretch_width",
        ),
        "heat_simulate": pn.widgets.Button(
            name="Run scenario analysis",
            design=Bootstrap,
            button_type="primary",
            disabled=False,
            icon="caret-right",
        ),
        "visualize_simulation": pn.widgets.Button(
            name="Show scenario analysis results",
            design=Bootstrap,
            button_type="light",
            disabled=True,
            icon="caret-right",
            sizing_mode="stretch_width",
        ),
    }


def create_text_inputs():
    unique_id = str(uuid.uuid4())
    return {
        "project_name": pn.widgets.TextInput(
            name="Project name",
            placeholder="Enter project name...",
            align="start",
            width=250,
            value=unique_id,
        )
    }


def create_radio_groups():
    return {
        "project_type": pn.widgets.RadioBoxGroup(
            name="Project_type",
            options=["New project", "Existing project"],
            inline=True,
            align="start",
            design=Material,
        ),
        "nbs_options": pn.widgets.RadioBoxGroup(
            name="nbs_options",
            options=[
                "None",
                "Forest landscape restoration",
                "Forest protected areas",
                "Wetland conservation",
                "Riparian buffer",
                "Urban greening",
            ],
            align="start",
            inline=False,
            design=Material,
        ),
    }


def create_sliders():
    return {
        "temp": pn.widgets.FloatSlider(
            name="Change in daily temperature (degC)",
            start=0,
            end=4,
            step=0.5,
            value=1.5,
            disabled=True,
        ),
        "rsds": pn.widgets.FloatSlider(
            name="Change in daily solar radiation (%)",
            start=0,
            end=30,
            step=2,
            value=10,
            disabled=True,
        ),
        "ndvi": pn.widgets.FloatSlider(
            name="Expansion of urban greening areas (%)",
            start=-30,
            end=30,
            step=2,
            value=10,
            disabled=True,
        ),
        "peak_rf": pn.widgets.FloatSlider(
            name="Change in frequency of heavy rainfall days (%)",
            start=-50,
            end=50,
            step=5,
            value=10,
            disabled=True,
        ),
        "dry_days": pn.widgets.FloatSlider(
            name="Change in frequency of dry days (%)",
            start=-50,
            end=50,
            step=5,
            value=10,
            disabled=True,
        ),
    }


def create_statictext():
    return {
        "roi": pn.widgets.StaticText(design=Material),
    }


buttons = create_buttons()
text_inputs = create_text_inputs()
radio_groups = create_radio_groups()
sliders = create_sliders()
roi_text = create_statictext()


def set_widget_state(widgets, state_config):
    for widget_name, config in state_config.items():
        widget = widgets.get(widget_name)
        if widget:
            widget.disabled = config.get("disabled", widget.disabled)


def set_button_state(buttons, state_config):
    for button_name, config in state_config.items():
        button = buttons.get(button_name)
        if button:
            button.disabled = config.get("disabled", button.disabled)
            button.button_type = config.get("button_type", button.button_type)


# ====================================================================================================================================================================
def update_buttons(event):
    nbs_value = radio_groups["nbs_options"].value

    # Define button states based on nbs_value
    if nbs_value not in ["None", "Urban greening"]:
        set_button_state(
            buttons,
            {
                "evaluate": {"disabled": False, "button_type": "primary"},
                "simulate": {"disabled": False, "button_type": "primary"},
                "visualize_simulation": {
                    "disabled": False,
                    "button_type": "primary",
                },  # Only activate after simulation
            },
        )
        set_widget_state(
            sliders,
            {
                "temp": {"disabled": False},
                "peak_rf": {"disabled": False},
                "dry_days": {"disabled": False},
                "rsds": {"disabled": True},
                "ndvi": {"disabled": True},
            },
        )
        roi_text["roi"].value = (
            "Double click on a desired catchment area (blue polygon) to select it as study area."
        )

    elif nbs_value == "Urban greening":
        set_button_state(
            buttons,
            {
                "evaluate": {"disabled": False, "button_type": "primary"},
                "simulate": {"disabled": False, "button_type": "primary"},
                "visualize_simulation": {
                    "disabled": False,
                    "button_type": "primary",
                },  # Only activate after simulation
            },
        )
        set_widget_state(
            sliders,
            {
                "temp": {"disabled": False},
                "peak_rf": {"disabled": True},
                "dry_days": {"disabled": True},
                "rsds": {"disabled": False},
                "ndvi": {"disabled": False},
            },
        )
        roi_text["roi"].value = (
            "Zoom in to a desried city or urban area and draw a polygon around it to select it as study area."
        )

    else:  # Default case for 'None' or other unhandled options
        set_button_state(
            buttons,
            {
                "evaluate": {"disabled": True, "button_type": "light"},
                "simulate": {"disabled": True, "button_type": "light"},
                "visualize_simulation": {"disabled": True, "button_type": "light"},
            },
        )
        set_widget_state(
            sliders,
            {
                "temp": {"disabled": True},
                "peak_rf": {"disabled": True},
                "dry_days": {"disabled": True},
                "rsds": {"disabled": True},
                "ndvi": {"disabled": True},
            },
        )


radio_groups["nbs_options"].param.watch(update_buttons, "value")


# ====================================================================================================================================================================
def create_panels(buttons, radio_groups, text_inputs):
    wc1 = pn.Card(
        pn.WidgetBox(
            radio_groups["nbs_options"], width=300, css_classes=["bk-no-border"]
        ),
        header="### 1. Select a nature-based solution",
        collapsed=False,
        design=Material,
        width=320,
    )
    wc2 = pn.Card(
        pn.WidgetBox(buttons["evaluate"], width=320, css_classes=["bk-no-border"]),
        header="### 2. Show model reliability",
        collapsed=True,
        design=Material,
        width=320,
    )
    wc3 = pn.Card(
        sliders["peak_rf"],
        sliders["dry_days"],
        sliders["temp"],
        sliders["rsds"],
        sliders["ndvi"],
        buttons["simulate"],
        buttons["visualize_simulation"],
        header="### 4. Assess NbS effectiveness",
        collapsed=False,
        design=Material,
        width=320,
    )
    wc4 = pn.Card(
        roi_text["roi"],
        header="### 3. Select a study area",
        collapsed=False,
        design=Material,
        width=320,
    )
    return wc1, wc2, wc3, wc4


# ====================================================================================================================================================================

# define styles
# Custom CSS for removing the border and padding
wb_css = """
.bk-no-border .bk-widget-box {
    border: none;
    background-color: #F8F8F8;
    padding: 0px;
    box-shadow: none;
}
"""

land_cover_alert = """
<div style="color: orange;">
    <i class="fas fa-info-circle"></i> <strong>Downloading land cover data.</strong>
    <hr>
    Land cover data from the European Space Agency - WorldCover product 
    (<a href="https://esa-worldcover.org/en" target="_blank">https://esa-worldcover.org/en</a>) 
    is downloaded into the directory f'projects/{text_input.value}/scratch'.
</div>
"""

soil_alert = """
<div style="color: orange;">
    <i class="fas fa-info-circle"></i> <strong>Preprocessing hydrologic soil group data.</strong>
    <hr>
    This data were developed to support USDA-based curve-number runoff modeling at regional and continental scales.
    Classification of HSGs was derived from soil texture classes and depth to bedrock provided by the Food and Agriculture Organization soilGrids250m system. 
    The data is downloaded into the directory f'projects/{text_input.value}/land_cover'.
</div>
"""

wetland_alert = open("styles/wetland_alert.txt", "r").read()
protected_area_alert = open("styles/protected_areas.txt", "r").read()
forest_restoration_alert = open("styles/forest_restoration_alert.txt", "r").read()
intro_text = open("styles/intro_alert.txt", "r").read()

intro_alert = pn.pane.HTML(intro_text)

status_alert = pn.pane.Alert(
    """
<div style="color: dark green;">
    <i class="fas fa-info-circle"></i> <strong>NBS Toolbox Infobox</strong>
    <hr>
    \n\nReady to explore and deploy this toolbox for your projects? \n\nDynamic messages will be provided in this infobox to guide you through the use of this toolbox and provide status updates of running processes. \n\n To begin select a nature-based solution from the provided options.
</div>
""",
    alert_type="info",
)


existing_project_message = """
<div style="color: dark red;">
    <i class="fas fa-info-circle"></i> Project already exists! No need to draw a rectangle on the map around the catchment of interest.
</div>
"""

styles = {"background-color": "#F8F8F8", "padding": "10px", "overflow": "hidden"}

styles1 = {
    "background-color": "#00A170",
    "color": "white",
    "font-family": "Arial, sans-serif",
}

styles2 = {
    "background-color": "#F8F8F8",
    "color": "white",
    "font-family": "Arial, sans-serif",
    "overflow": "auto",
}

footer_html = """
<footer style="background-color: #F6F6F6; padding: 10px; text-align: center; font-family: Arial, sans-serif; color: #333;">
    <p>
        The development of this toolbox was funded by WUR Investment theme, Data-Driven Discoveries in a Changing Climate.
        For more information about the toolbox contact Confidence Duku at
        <a href="mailto:confidence.duku@wur.nl" style="color: #1a73e8; text-decoration: none;">confidence.duku@wur.nl</a>.
    </p>
</footer>
"""
urban_green_image = pn.pane.Image(
    "https://static.vecteezy.com/system/resources/thumbnails/044/751/695/small_2x/public-garden-in-the-city-panoramic-view-of-city-park-with-green-trees-grass-and-cityscape-on-background-illustration-cartoon-landscape-with-empty-park-and-town-buildings-on-skyline-vector.jpg",
    sizing_mode="stretch_width",
    caption="Photo credit:",
)

wetland_image = pn.pane.Image(
    "https://www.undp.org/sites/g/files/zskgke326/files/migration/zw/UNDP_ZW_AccLab_Wetlands_3.jpg",
    sizing_mode="stretch_width",
    caption="Photo credit: UNDP",
)

# intro_image = pn.pane.Image('https://cdn.unenvironment.org/s3fs-public/inline-images/Implementation.png?VersionId=null', sizing_mode='stretch_width', caption='Photo credit: UNEP Adaptation Gap Report, 2020')

intro_image = pn.pane.Image(
    "https://ensia.com/wp-content/uploads/2022/03/Voices_nature-positive_main-920x460.jpg",
    sizing_mode="stretch_width",
    caption="Photo credit:  Kelsey Kin",
)

restoration_image = pn.pane.Image(
    "https://www.iucn.org/sites/default/files/content/images/2021/ppt_background2.png",
    caption="Photo credit: IUCN stock images",
    sizing_mode="stretch_width",
)

protected_areas_image = pn.pane.Image(
    "https://ik.imagekit.io/gn7hrls3k/banner_ZMmPXBKdKH.webp",
    caption="Photo credit:",
    sizing_mode="stretch_width",
)

riparian_image = pn.pane.Image(
    "images/riparian_image.jpg",
    caption="Photo credit: Jessica Puglisi/Open Space Institute",
    sizing_mode="stretch_width",
)

intro = pn.Column(intro_alert, intro_image)

header = pn.Card(
    pn.pane.HTML(
        """
<h1>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;EcoEffect: providing data-driven insights on the effectiveness of nature-based solutions</h1>
""",
        styles=styles1,
    ),
    sizing_mode="stretch_width",
    styles=styles1,
    hide_header=True,
    min_height=80,
)


# =====================================================================================================================================================
# code logic and project setup
def create_project_structure(project_name, nbs_type):
    dirs_to_create = PROJECT_DIR_STRUCTURE.get(
        nbs_type, PROJECT_DIR_STRUCTURE["default"]
    )
    for dir_path in dirs_to_create:
        os.makedirs(f"projects/{project_name}/{dir_path}", exist_ok=True)


# code logics
def check_existing_file(file_path):
    directory, filename = os.path.split(file_path)
    if os.path.exists(file_path):
        return True
    else:
        return False


def simulate_out(event):
    m3 = leafmap.Map(height=650)
    m3.split_map(left_layer="TERRAIN", right_layer="OpenTopoMap")

    m4 = leafmap.Map(height=650, center=[39.4948, -108.5492], zoom=12)
    url = (
        "https://github.com/opengeos/data/releases/download/raster/Libya-2023-07-01.tif"
    )
    url2 = (
        "https://github.com/opengeos/data/releases/download/raster/Libya-2023-09-13.tif"
    )
    m4.split_map(url, url2)

    main_page[0] = pn.Tabs(
        ("Flood risk", pn.Card(m3, hide_header=True, sizing_mode="stretch_width")),
        (
            "Soil erosion risk",
            pn.Card(m4, hide_header=True, sizing_mode="stretch_width"),
        ),
        ("Drought risk", pn.Card(m3, hide_header=True, sizing_mode="stretch_width")),
        width_policy="fit",
    )

    # Remove the right sidebar and expand the main_page
    if len(main) == 3:
        main.pop(-1)  # Remove right_sidebar
        main[-1].width = None  # Reset width to allow it to expand
        main[-1].styles = {}  # Reset any custom styles if needed

    # Ensure main layout updates correctly
    main[-1].sizing_mode = "stretch_both"  # Ensure it expands to fill available space
    # right_sidebar.clear()
    # main.pop(2)


clicked_bbox = None
gdf = gpd.read_file("common_data/hydrosheds_l4_wa.shp")


def select_polygon(**kwargs):
    global clicked_bbox
    if kwargs.get("type") == "click":
        latlon = kwargs.get("coordinates")
        geometry = Point(latlon[::-1])
        selected = gdf[gdf.intersects(geometry)]
        setattr(m, "zoom_to_layer", False)
        if len(selected) > 0:
            clicked_bbox = tuple(selected.geometry.bounds.iloc[0])
        else:
            clicked_bbox = None


def nbs_status_alert():
    if radio_groups["nbs_options"].value == "Forest landscape restoration":
        intro_alert.object = forest_restoration_alert
        intro[1] = restoration_image
    elif radio_groups["nbs_options"].value == "Forest protected areas":
        intro_alert.object = protected_area_alert
        intro[1] = protected_areas_image
    elif radio_groups["nbs_options"].value == "Wetland conservation":
        intro_alert.object = wetland_alert
        intro[1] = wetland_image
    elif radio_groups["nbs_options"].value == "Riparian buffer":
        intro_alert.object = wetland_alert
        intro[1] = riparian_image
    elif radio_groups["nbs_options"].value == "Urban greening":
        intro[1] = urban_green_image


def analyse_scenario(event):
    proj_name = str(uuid.uuid4())
    nbs_type = radio_groups["nbs_options"].value
    create_project_structure(proj_name, nbs_type)

    if nbs_type not in ["Urban greening", "None"]:
        from hydro.hydro import DeepSTRMM

        ds = DeepSTRMM(proj_name, clicked_bbox)


# intialise map elements
bounds = [[0.349392, -4.615363], [10.399066, 4.377311]]
# bounds = [[11.0, -1.5], [15.0, 2.5]]  # Replace with actual bounds
m = leafmap.Map(center=[12.580827, 0.727553], zoom=1, draw_control=False, height=700)
m.fit_bounds(bounds)
m.add_basemap()
folium_pane = pn.Card(pn.panel(m), hide_header=True, sizing_mode="stretch_both")
main_page = pn.Column(folium_pane, sizing_mode="stretch_both")

m2 = leafmap.Map(center=[12.580827, 0.727553], zoom=7, height=700)
# bounds = [[11.0, -1.5], [15.0, 2.5]]  # Replace with actual bounds

mt = leafmap.Map(center=[12.580827, 0.727553], zoom=5, height=700)
mt.add_basemap()

# Layer configurations
layer_configs = {
    "Forest landscape restoration": [
        (
            "common_data/hydrosheds2.geojson",
            "Sub-basins",
            {"weight": 0.5, "color": "blue"},
        ),
        (
            "common_data/wri_reforestation_glo.tif",
            "Reforestation opportunities",
            "raster",
        ),
    ],
    "Forest protected areas": [
        (
            "common_data/hydrosheds2.geojson",
            "Sub-basins",
            {"weight": 0.5, "color": "blue"},
        ),
        ("common_data/wpda_forests.tif", "Protected forests", "raster"),
    ],
    "Wetland conservation": [
        (
            "common_data/hydrosheds2.geojson",
            "Sub-basins",
            {"weight": 0.5, "color": "blue"},
        ),
        ("common_data/wpda_wetlands.tif", "wetlands", "raster"),
    ],
    "Riparian buffer": [
        (
            "common_data/hydrosheds2.geojson",
            "Sub-basins",
            {"weight": 0.5, "color": "blue"},
        ),
    ],
    "Urban greening": [],
    # Add other configurations if needed
}


def update_map(nbs_value):
    m2.clear_layers()  # Clear existing layers
    m2.add_basemap()  # Re-add basemap

    layers = layer_configs.get(nbs_value)
    if layers:
        for layer in layers:
            if layer[2] == "raster":
                m2.add_raster(layer[0], layer_name=layer[1], zoom_to_layer=True)
                m2.zoom = 7
                m2.center = [12.580827, 0.727553]
            else:
                m2.add_geojson(
                    layer[0],
                    layer_name=layer[1],
                    info_mode=None,
                    zoom_to_layer=True,
                    style=layer[2],
                )


def update_dashboard(event):
    # project_name = text_inputs['project_name'].value
    nbs_value = radio_groups["nbs_options"].value
    project_value = radio_groups["project_type"].value

    # Update the map based on the selected nbs_value
    if nbs_value in layer_configs:
        update_map(nbs_value)
        main_page[0] = pn.panel(pn.Card(m2, hide_header=True))
        if len(main) <= 2:
            main_page[0].width = 700
            right_sidebar.max_width = 450
            main.append(right_sidebar)
        nbs_status_alert()
        # buttons['evaluate'].enabled = nbs_value == 'Urban greening'
        # buttons['evaluate'].button_type = 'primary' if nbs_value != 'Urban greening' else 'default'
    else:
        buttons["evaluate"].disabled = True


# ==============================================================================================================================================
# set callbacks
m2.on_interaction(select_polygon)
buttons["visualize_simulation"].on_click(simulate_out)
# buttons['download'].on_click(download_inputs)
buttons["simulate"].on_click(analyse_scenario)
radio_groups["nbs_options"].param.watch(update_dashboard, "value")
# radio_groups['nbs_options'].param.watch(nbs_status_alert, 'value')
# radio_groups['nbs_options'].param.watch(create_project_structure, 'value')
# nbs_radio_group.param.watch(nbs_maps, 'value')

radio_groups["project_type"].param.watch(update_dashboard, "value")
text_inputs["project_name"].param.watch(update_dashboard, "value")

text_inputs["project_name"].param.watch(update_buttons, "value")
radio_groups["nbs_options"].param.watch(update_buttons, "value")
radio_groups["project_type"].param.watch(update_buttons, "value")

# =========================================================================================================================================
pn.extension("floatpanel", "ipywidgets", design="material")
wc1, wc2, wc3, wc4 = create_panels(buttons, radio_groups, text_inputs)
left_sidebar = pn.Column(wc1, wc2, wc4, wc3, max_width=330, scroll=True, styles=styles2)
# right_sidebar = pn.Column(pn.Card(status_alert, hide_header=True, max_width=450), pn.Card(intro, max_width=450, hide_header=True, sizing_mode='stretch_height'), max_width=450)
right_sidebar = pn.Card(
    intro, max_width=450, hide_header=True, sizing_mode="stretch_height"
)
main = pn.Row(left_sidebar, main_page, right_sidebar, styles=styles)
new_template = pn.Column(
    header, main, styles=styles, scroll=False, sizing_mode="stretch_height"
).servable()

# Serve the app via Python script
if __name__ == "__main__":
    pn.serve(new_template, title="EcoEffect Dashboard", port=8000)
