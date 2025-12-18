#####   1. Getting and preprocessing input data

If you have previously executed this step and downloaded all input data you can skip this step and proceed directly to step 2. However, if you are not sure, run step 1 and the model will either confirm that a specific data has aready been downloaded and subsequently viualize it or it will proceeed to download the data if it is not available or prior download was incomplete.


```python
%cd /home/WUR/duku002/Scripts/drought_floods/vscode
```

    /home/WUR/duku002/Scripts/drought_floods/vscode



```python
working_dir='/lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger'
study_area='/home/WUR/duku002/Scripts/NBAT/hydro/common_data/niger.shp'
```


```python

# download and preprocess MODIS vegetation continuous fields from Google Earth Engine Data catalog

from bakaano.tree_cover import TreeCover
vf = TreeCover(
    working_dir=working_dir, 
    study_area=study_area, 
    start_date='2001-01-01', 
    end_date='2020-12-31'
)
vf.get_tree_cover_data()
vf.plot_tree_cover(variable='tree_cover') # options for plot are 'tree_cover' and 'herb_cover'
```

         - Tree cover data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/vcf/mean_tree_cover.tif; skipping download.
         - Tree cover data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/vcf/mean_tree_cover.tif; skipping preprocessing.



    
![png](quick_start_files/quick_start_3_1.png)
    



```python
# download and preprocess MODIS NDVI data from Google Earth Engine Data catalog

from bakaano.ndvi import NDVI
nd = NDVI(
    working_dir=working_dir, 
    study_area=study_area, 
    start_date='2001-01-01', 
    end_date='2010-12-31'
)
nd.get_ndvi_data()
nd.plot_ndvi(interval_num=10)  # because NDVI is in 16-day interval the 'interval_num' represents a 16-day period. 
                               #Hence 0 is the first 16 day period
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



         - NDVI data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/ndvi/daily_ndvi_climatology.pkl; skipping download.
         - NDVI data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/ndvi/daily_ndvi_climatology.pkl; skipping preprocessing.



    
![png](quick_start_files/quick_start_4_2.png)
    



```python
# Get elevation data

from bakaano.dem import DEM
dd = DEM(
    working_dir=working_dir, 
    study_area=study_area, 
    local_data=False, 
    local_data_path=None
)
dd.get_dem_data()
dd.plot_dem()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



         - DEM data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/elevation; skipping download.



    
![png](quick_start_files/quick_start_5_2.png)
    



```python
# Get soil data

from bakaano.soil import Soil
sgd = Soil(
    working_dir=working_dir, 
    study_area=study_area
)
sgd.get_soil_data()
sgd.plot_soil(variable='wilting_point')  #options are 'wilting_point', 'saturation_point' and 'available_water_content'
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



         - Soil data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/soil; skipping download.



    
![png](quick_start_files/quick_start_6_2.png)
    



```python
#  Get alpha earth satellite embedding dataset

from bakaano.alpha_earth import AlphaEarth
dd = AlphaEarth(
    working_dir=working_dir, 
    study_area=study_area,
    start_date='2013-01-01', 
    end_date = '2024-01-01',
)
dd.get_alpha_earth()
dd.plot_alpha_earth('A35') #Band options are A00 to A63
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    ✓ All 64 AlphaEarth bands already downloaded. Skipping.



    
![png](quick_start_files/quick_start_7_2.png)
    



```python
#get meteo

from bakaano.meteo import Meteo
cd = Meteo(
    working_dir=working_dir, 
    study_area=study_area, 
    start_date='2001-01-01', 
    end_date='2010-12-31',
    local_data=False, 
    data_source='ERA5'
)
cd.plot_meteo(variable='tasmin', date='2006-12-01') # variable options are 'tmean', 'precip', 'tasmax', 'tasmin'
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



         - ERA5 Land daily data already exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/era5_land; skipping download.



    
![png](quick_start_files/quick_start_8_2.png)
    


#####   2. Computing runoff and routing to river network


```python

from bakaano.veget import VegET
vg = VegET(
    working_dir=working_dir, 
    study_area=study_area,
    start_date='2001-01-01', 
    end_date='2010-12-31',
    climate_data_source='ERA5',
    routing_method='mfd'
)
vg.compute_veget_runoff_route_flow()
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    Routed runoff data exists in /lustre/backup/WUR/ESG/duku002/Drought-Flood-Cascade/niger/runoff_output/wacc_sparse_arrays.pkl. Skipping processing



```python
#visualize routed runoff data

from bakaano.plot_runoff import RoutedRunoff
rr = RoutedRunoff(
    working_dir=working_dir, 
    study_area=study_area
)
rr.map_routed_runoff(date='2020-09-03', vmax=6) #output values have been log transformed for better visualization
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




    
![png](quick_start_files/quick_start_11_1.png)
    


#####   3. Explore input data, river networks and hydrological stations interactively


```python
from bakaano.runner import BakaanoHydro
bk = BakaanoHydro(
    working_dir=working_dir, 
    study_area=study_area,
    climate_data_source='ERA5'
)
bk.explore_data_interactively('1981-01-01', '2016-12-31', '/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-africa-south-america.nc')
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    2025-12-18 13:11:20.637642: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    E0000 00:00:1766059880.648137   29599 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
    E0000 00:00:1766059880.651224   29599 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered





<iframe srcdoc="&lt;!DOCTYPE html&gt;
&lt;html&gt;
&lt;head&gt;

    &lt;meta http-equiv=&quot;content-type&quot; content=&quot;text/html; charset=UTF-8&quot; /&gt;

        &lt;script&gt;
            L_NO_TOUCH = false;
            L_DISABLE_3D = false;
        &lt;/script&gt;

    &lt;style&gt;html, body {width: 100%;height: 100%;margin: 0;padding: 0;}&lt;/style&gt;
    &lt;style&gt;#map {position:absolute;top:0;bottom:0;right:0;left:0;}&lt;/style&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://code.jquery.com/jquery-3.7.1.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/js/bootstrap.bundle.min.js&quot;&gt;&lt;/script&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/bootstrap@5.2.2/dist/css/bootstrap.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://netdna.bootstrapcdn.com/bootstrap/3.0.0/css/bootstrap-glyphicons.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.2.0/css/all.min.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css&quot;/&gt;

            &lt;meta name=&quot;viewport&quot; content=&quot;width=device-width,
                initial-scale=1.0, maximum-scale=1.0, user-scalable=no&quot; /&gt;
            &lt;style&gt;
                #map_b85d2c3c819d33f59d25ee97b1f49321 {
                    position: relative;
                    width: 100.0%;
                    height: 100.0%;
                    left: 0.0%;
                    top: 0.0%;
                }
                .leaflet-container { font-size: 1rem; }
            &lt;/style&gt;

    &lt;script src=&quot;https://cdn.jsdelivr.net/npm/leaflet.fullscreen@3.0.0/Control.FullScreen.min.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdn.jsdelivr.net/npm/leaflet.fullscreen@3.0.0/Control.FullScreen.css&quot;/&gt;
    &lt;script src=&quot;https://unpkg.com/leaflet-control-geocoder/dist/Control.Geocoder.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://unpkg.com/leaflet-control-geocoder/dist/Control.Geocoder.css&quot;/&gt;
    &lt;script src=&quot;https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/leaflet.markercluster.js&quot;&gt;&lt;/script&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.css&quot;/&gt;
    &lt;link rel=&quot;stylesheet&quot; href=&quot;https://cdnjs.cloudflare.com/ajax/libs/leaflet.markercluster/1.1.0/MarkerCluster.Default.css&quot;/&gt;
&lt;/head&gt;
&lt;body&gt;


            &lt;div class=&quot;folium-map&quot; id=&quot;map_b85d2c3c819d33f59d25ee97b1f49321&quot; &gt;&lt;/div&gt;

&lt;/body&gt;
&lt;script&gt;


            var map_b85d2c3c819d33f59d25ee97b1f49321 = L.map(
                &quot;map_b85d2c3c819d33f59d25ee97b1f49321&quot;,
                {
                    center: [20.0, 0.0],
                    crs: L.CRS.EPSG3857,
                    ...{
  &quot;zoom&quot;: 2,
  &quot;zoomControl&quot;: true,
  &quot;preferCanvas&quot;: false,
  &quot;drawExport&quot;: false,
  &quot;layersControl&quot;: true,
}

                }
            );
            L.control.scale().addTo(map_b85d2c3c819d33f59d25ee97b1f49321);





            var tile_layer_a6c1f35601ac2f9e3811d50c058f037f = L.tileLayer(
                &quot;https://tile.openstreetmap.org/{z}/{x}/{y}.png&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 24,
  &quot;maxNativeZoom&quot;: 24,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;\u0026copy; \u003ca href=\&quot;https://www.openstreetmap.org/copyright\&quot;\u003eOpenStreetMap\u003c/a\u003e contributors&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 1,
}

            );


            tile_layer_a6c1f35601ac2f9e3811d50c058f037f.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            L.control.fullscreen(
                {
  &quot;position&quot;: &quot;topleft&quot;,
  &quot;title&quot;: &quot;Full Screen&quot;,
  &quot;titleCancel&quot;: &quot;Exit Full Screen&quot;,
  &quot;forceSeparateButton&quot;: false,
}
            ).addTo(map_b85d2c3c819d33f59d25ee97b1f49321);



            var geocoderOpts_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f = {
  &quot;collapsed&quot;: true,
  &quot;position&quot;: &quot;topleft&quot;,
  &quot;defaultMarkGeocode&quot;: true,
  &quot;zoom&quot;: 11,
  &quot;provider&quot;: &quot;nominatim&quot;,
  &quot;providerOptions&quot;: {
},
};

            // note: geocoder name should start with lowercase
            var geocoderName_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f = geocoderOpts_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f[&quot;provider&quot;];

            var customGeocoder_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f = L.Control.Geocoder[ geocoderName_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f ](
                geocoderOpts_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f[&#x27;providerOptions&#x27;]
            );
            geocoderOpts_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f[&quot;geocoder&quot;] = customGeocoder_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f;

            L.Control.geocoder(
                geocoderOpts_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f
            ).on(&#x27;markgeocode&#x27;, function(e) {
                var zoom = geocoderOpts_geocoder_42f388f3f13a87d98eefbfd26ac4ba7f[&#x27;zoom&#x27;] || map_b85d2c3c819d33f59d25ee97b1f49321.getZoom();
                map_b85d2c3c819d33f59d25ee97b1f49321.setView(e.geocode.center, zoom);
            }).addTo(map_b85d2c3c819d33f59d25ee97b1f49321);



            map_b85d2c3c819d33f59d25ee97b1f49321.fitBounds(
                [[20, 0], [20, 0]],
                {&quot;maxZoom&quot;: 2}
            );


            var tile_layer_0401fe79e3b17c0ea9fb52edd4b15068 = L.tileLayer(
                &quot;/user/duku002//proxy/33591/api/tiles/{z}/{x}/{y}.png?\u0026filename=%2Flustre%2Fbackup%2FWUR%2FESG%2Fduku002%2FDrought-Flood-Cascade%2Fniger%2Felevation%2Fdem_clipped.tif\u0026colormap=gist_ncar&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 30,
  &quot;maxNativeZoom&quot;: 30,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;Raster file served by \u003ca href=\u0027https://github.com/banesullivan/localtileserver\u0027 target=\u0027_blank\u0027\u003elocaltileserver\u003c/a\u003e.&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 0.6,
  &quot;bounds&quot;: [[4.391667, -11.591667], [23.958333, 15.858333]],
  &quot;zoomToLayer&quot;: true,
}

            );


            tile_layer_0401fe79e3b17c0ea9fb52edd4b15068.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            map_b85d2c3c819d33f59d25ee97b1f49321.fitBounds(
                [[4.391667, -11.591667], [23.958333, 15.858333]],
                {}
            );


            var tile_layer_9c50beee1fd49223f644ea8a918cbf30 = L.tileLayer(
                &quot;/user/duku002//proxy/33591/api/tiles/{z}/{x}/{y}.png?\u0026filename=%2Flustre%2Fbackup%2FWUR%2FESG%2Fduku002%2FDrought-Flood-Cascade%2Fniger%2Fsoil%2Fclipped_AWCh3_M_sl6_1km_ll.tif\u0026colormap=terrain\u0026vmax=10&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 30,
  &quot;maxNativeZoom&quot;: 30,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;Raster file served by \u003ca href=\u0027https://github.com/banesullivan/localtileserver\u0027 target=\u0027_blank\u0027\u003elocaltileserver\u003c/a\u003e.&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 0.75,
  &quot;bounds&quot;: [[4.390834, -11.591667], [23.957501, 15.858333]],
  &quot;zoomToLayer&quot;: true,
  &quot;visible&quot;: false,
}

            );


            tile_layer_9c50beee1fd49223f644ea8a918cbf30.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            map_b85d2c3c819d33f59d25ee97b1f49321.fitBounds(
                [[4.390834, -11.591667], [23.957501, 15.858333]],
                {}
            );


            var tile_layer_4b7566c2434891991626134f7c96d735 = L.tileLayer(
                &quot;/user/duku002//proxy/33591/api/tiles/{z}/{x}/{y}.png?\u0026filename=%2Flustre%2Fbackup%2FWUR%2FESG%2Fduku002%2FDrought-Flood-Cascade%2Fniger%2Fvcf%2Fmean_tree_cover.tif\u0026colormap=viridis_r\u0026vmax=70&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 30,
  &quot;maxNativeZoom&quot;: 30,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;Raster file served by \u003ca href=\u0027https://github.com/banesullivan/localtileserver\u0027 target=\u0027_blank\u0027\u003elocaltileserver\u003c/a\u003e.&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 0.75,
  &quot;bounds&quot;: [[4.383779, -11.588267], [23.958069, 15.864248]],
  &quot;zoomToLayer&quot;: true,
  &quot;visible&quot;: false,
}

            );


            tile_layer_4b7566c2434891991626134f7c96d735.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            map_b85d2c3c819d33f59d25ee97b1f49321.fitBounds(
                [[4.383779, -11.588267], [23.958069, 15.864248]],
                {}
            );


            var tile_layer_f2cc6261c50d7f64826a60bd81da39c0 = L.tileLayer(
                &quot;/user/duku002//proxy/33591/api/tiles/{z}/{x}/{y}.png?\u0026filename=%2Flustre%2Fbackup%2FWUR%2FESG%2Fduku002%2FDrought-Flood-Cascade%2Fniger%2Felevation%2Fslope_clipped.tif\u0026colormap=gist_ncar\u0026vmax=20&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 30,
  &quot;maxNativeZoom&quot;: 30,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;Raster file served by \u003ca href=\u0027https://github.com/banesullivan/localtileserver\u0027 target=\u0027_blank\u0027\u003elocaltileserver\u003c/a\u003e.&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 0.75,
  &quot;bounds&quot;: [[4.391667, -11.591667], [23.958333, 15.858333]],
  &quot;zoomToLayer&quot;: true,
  &quot;visible&quot;: false,
}

            );


            tile_layer_f2cc6261c50d7f64826a60bd81da39c0.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            map_b85d2c3c819d33f59d25ee97b1f49321.fitBounds(
                [[4.391667, -11.591667], [23.958333, 15.858333]],
                {}
            );


            var tile_layer_4c542af7e89a3ccc700d66b07892850f = L.tileLayer(
                &quot;/user/duku002//proxy/33591/api/tiles/{z}/{x}/{y}.png?\u0026filename=%2Flustre%2Fbackup%2FWUR%2FESG%2Fduku002%2FDrought-Flood-Cascade%2Fniger%2Fscratch%2Friver_network.tif\u0026colormap=viridis\u0026vmax=80853.8689214557&quot;,
                {
  &quot;minZoom&quot;: 0,
  &quot;maxZoom&quot;: 30,
  &quot;maxNativeZoom&quot;: 30,
  &quot;noWrap&quot;: false,
  &quot;attribution&quot;: &quot;Raster file served by \u003ca href=\u0027https://github.com/banesullivan/localtileserver\u0027 target=\u0027_blank\u0027\u003elocaltileserver\u003c/a\u003e.&quot;,
  &quot;subdomains&quot;: &quot;abc&quot;,
  &quot;detectRetina&quot;: false,
  &quot;tms&quot;: false,
  &quot;opacity&quot;: 0.9,
  &quot;bounds&quot;: [[4.391667, -11.591667], [23.958333, 15.858333]],
  &quot;zoomToLayer&quot;: true,
  &quot;visible&quot;: false,
}

            );


            tile_layer_4c542af7e89a3ccc700d66b07892850f.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            map_b85d2c3c819d33f59d25ee97b1f49321.fitBounds(
                [[4.391667, -11.591667], [23.958333, 15.858333]],
                {}
            );


            var marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae = L.markerClusterGroup(
                {
  &quot;color&quot;: &quot;brown&quot;,
  &quot;radius&quot;: 3,
  &quot;maxClusterRadius&quot;: 80,
}
            );


            var marker_ed30f8a5c45d6586813865597afeef43 = L.marker(
                [11.149999618530273, -8.550000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_c5235c311abe7f98a71d7b9c47012e45 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_260cfb63b558e86ebc3e74caa333dfa3 = $(`&lt;div id=&quot;html_260cfb63b558e86ebc3e74caa333dfa3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GUELELINKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.550000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.149999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_c5235c311abe7f98a71d7b9c47012e45.setContent(html_260cfb63b558e86ebc3e74caa333dfa3);



        marker_ed30f8a5c45d6586813865597afeef43.bindPopup(popup_c5235c311abe7f98a71d7b9c47012e45)
        ;




            var marker_7d1b54d75ecf28e2556dfb7e950017c3 = L.marker(
                [11.683300018310547, -8.66670036315918],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_adb1cd6b51270cd0d6792ec3969a8c47 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_f38f46e10d560966204dbb062d773c93 = $(`&lt;div id=&quot;html_f38f46e10d560966204dbb062d773c93&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: BANANKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 52.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.66670036315918&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.683300018310547&lt;br&gt;&lt;/div&gt;`)[0];
                popup_adb1cd6b51270cd0d6792ec3969a8c47.setContent(html_f38f46e10d560966204dbb062d773c93);



        marker_7d1b54d75ecf28e2556dfb7e950017c3.bindPopup(popup_adb1cd6b51270cd0d6792ec3969a8c47)
        ;




            var marker_79877a42713befc2bd4edf650922db6f = L.marker(
                [11.970000267028809, -8.229999542236328],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_93aefdea1d9a886ee722e3fb5fc54834 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_d2208e6214d648d0f9efdffa546686ac = $(`&lt;div id=&quot;html_d2208e6214d648d0f9efdffa546686ac&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GOUALA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.229999542236328&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.970000267028809&lt;br&gt;&lt;/div&gt;`)[0];
                popup_93aefdea1d9a886ee722e3fb5fc54834.setContent(html_d2208e6214d648d0f9efdffa546686ac);



        marker_79877a42713befc2bd4edf650922db6f.bindPopup(popup_93aefdea1d9a886ee722e3fb5fc54834)
        ;




            var marker_65bc7f61d2b0aa3c5513906c50d8403c = L.marker(
                [11.579999923706055, -8.170000076293945],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_f2048334bf9b6aa80adfad8bff29b4f9 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_a3870f0548664f43d03364d790a0fd57 = $(`&lt;div id=&quot;html_a3870f0548664f43d03364d790a0fd57&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: SELINGUE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.170000076293945&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.579999923706055&lt;br&gt;&lt;/div&gt;`)[0];
                popup_f2048334bf9b6aa80adfad8bff29b4f9.setContent(html_a3870f0548664f43d03364d790a0fd57);



        marker_65bc7f61d2b0aa3c5513906c50d8403c.bindPopup(popup_f2048334bf9b6aa80adfad8bff29b4f9)
        ;




            var marker_d5f0245129795245cd9a118883899c7f = L.marker(
                [11.130000114440918, -8.199999809265137],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_35860e027c4fbf359ff0843390e7c1ff = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_0fdaa5ba242cb39629e6423dec82398a = $(`&lt;div id=&quot;html_0fdaa5ba242cb39629e6423dec82398a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: YANFOLILA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.199999809265137&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.130000114440918&lt;br&gt;&lt;/div&gt;`)[0];
                popup_35860e027c4fbf359ff0843390e7c1ff.setContent(html_0fdaa5ba242cb39629e6423dec82398a);



        marker_d5f0245129795245cd9a118883899c7f.bindPopup(popup_35860e027c4fbf359ff0843390e7c1ff)
        ;




            var marker_749ab203832ace08d0cd5a363a63f0ef = L.marker(
                [10.800000190734863, -7.670000076293945],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_ce1aa22915597c6477549373a2faaa37 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_3f4c365624425360cb285d615872fa3f = $(`&lt;div id=&quot;html_3f4c365624425360cb285d615872fa3f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MADINA DIASSA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 89.6&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.670000076293945&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.800000190734863&lt;br&gt;&lt;/div&gt;`)[0];
                popup_ce1aa22915597c6477549373a2faaa37.setContent(html_3f4c365624425360cb285d615872fa3f);



        marker_749ab203832ace08d0cd5a363a63f0ef.bindPopup(popup_ce1aa22915597c6477549373a2faaa37)
        ;




            var marker_7bba01445381eda6531824a0a493c253 = L.marker(
                [12.866700172424316, -7.550000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_0b4c9bed42045247bbd1f7b0452c7144 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_cf9efd971366845f6a198afbfafd1edb = $(`&lt;div id=&quot;html_cf9efd971366845f6a198afbfafd1edb&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOULIKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 32.6&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.550000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.866700172424316&lt;br&gt;&lt;/div&gt;`)[0];
                popup_0b4c9bed42045247bbd1f7b0452c7144.setContent(html_cf9efd971366845f6a198afbfafd1edb);



        marker_7bba01445381eda6531824a0a493c253.bindPopup(popup_0b4c9bed42045247bbd1f7b0452c7144)
        ;




            var marker_ec03717d122aa3d4879653f0724b8d4e = L.marker(
                [11.399999618530273, -7.449999809265137],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_812234dd1d779ed9453ad55933912e1c = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_5f22273da4af8a5d8be0fa377a484b7a = $(`&lt;div id=&quot;html_5f22273da4af8a5d8be0fa377a484b7a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: BOUGOUNI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 88.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.449999809265137&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.399999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_812234dd1d779ed9453ad55933912e1c.setContent(html_5f22273da4af8a5d8be0fa377a484b7a);



        marker_ec03717d122aa3d4879653f0724b8d4e.bindPopup(popup_812234dd1d779ed9453ad55933912e1c)
        ;




            var marker_bfcf8eef0f3f066befeda2d269934a8b = L.marker(
                [11.069999694824219, -6.849999904632568],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_bbe16e8171f5304d6aabbcbc98dd3a77 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_3211d784f125c1c0dba83284dd2020b1 = $(`&lt;div id=&quot;html_3211d784f125c1c0dba83284dd2020b1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOLONDIEBA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 92.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.849999904632568&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.069999694824219&lt;br&gt;&lt;/div&gt;`)[0];
                popup_bbe16e8171f5304d6aabbcbc98dd3a77.setContent(html_3211d784f125c1c0dba83284dd2020b1);



        marker_bfcf8eef0f3f066befeda2d269934a8b.bindPopup(popup_bbe16e8171f5304d6aabbcbc98dd3a77)
        ;




            var marker_823fda726d2494ad8a5f61f05f2426b7 = L.marker(
                [12.520000457763672, -6.800000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_ec872ad566032283edc801cf99c5d80f = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_55760d0c35bc5b975114b110d21e026d = $(`&lt;div id=&quot;html_55760d0c35bc5b975114b110d21e026d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DIOILA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 89.1&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.800000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.520000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_ec872ad566032283edc801cf99c5d80f.setContent(html_55760d0c35bc5b975114b110d21e026d);



        marker_823fda726d2494ad8a5f61f05f2426b7.bindPopup(popup_ec872ad566032283edc801cf99c5d80f)
        ;




            var marker_44c004edd997f550c7e86c2931ee7ca3 = L.marker(
                [11.420000076293945, -6.570000171661377],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_5d9def87d29d6ec158f6b8c6e4f55b2c = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_66b46feb65af99e8587fb4a509522618 = $(`&lt;div id=&quot;html_66b46feb65af99e8587fb4a509522618&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: PANKOUROU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 88.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.570000171661377&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.420000076293945&lt;br&gt;&lt;/div&gt;`)[0];
                popup_5d9def87d29d6ec158f6b8c6e4f55b2c.setContent(html_66b46feb65af99e8587fb4a509522618);



        marker_44c004edd997f550c7e86c2931ee7ca3.bindPopup(popup_5d9def87d29d6ec158f6b8c6e4f55b2c)
        ;




            var marker_e2e9879fa837ca4c71f91251985ecde3 = L.marker(
                [13.716699600219727, -6.050000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_bfd83f7028b4278d0ac7f2bfbef2fc04 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_e5fe2358ea3aa43cced4b9dc39402280 = $(`&lt;div id=&quot;html_e5fe2358ea3aa43cced4b9dc39402280&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KIRANGO AVAL&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 55.3&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.050000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.716699600219727&lt;br&gt;&lt;/div&gt;`)[0];
                popup_bfd83f7028b4278d0ac7f2bfbef2fc04.setContent(html_e5fe2358ea3aa43cced4b9dc39402280);



        marker_e2e9879fa837ca4c71f91251985ecde3.bindPopup(popup_bfd83f7028b4278d0ac7f2bfbef2fc04)
        ;




            var marker_68dfa4e04359226b094ef67b74360ab0 = L.marker(
                [13.216699600219727, -5.900000095367432],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_f00559d8d31386c23530aa589b082e73 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_e32c43169eebf127b131e53433f1a82b = $(`&lt;div id=&quot;html_e32c43169eebf127b131e53433f1a82b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DOUNA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 46.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -5.900000095367432&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.216699600219727&lt;br&gt;&lt;/div&gt;`)[0];
                popup_f00559d8d31386c23530aa589b082e73.setContent(html_e32c43169eebf127b131e53433f1a82b);



        marker_68dfa4e04359226b094ef67b74360ab0.bindPopup(popup_f00559d8d31386c23530aa589b082e73)
        ;




            var marker_fb122ae9491728858c522354605ff291 = L.marker(
                [12.020000457763672, -5.679999828338623],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_8cff786d93b66cad6767312bfc91868e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_04265fb112091a51656ce31da4e7f9cf = $(`&lt;div id=&quot;html_04265fb112091a51656ce31da4e7f9cf&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOUORO 2&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 94.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -5.679999828338623&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.020000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_8cff786d93b66cad6767312bfc91868e.setContent(html_04265fb112091a51656ce31da4e7f9cf);



        marker_fb122ae9491728858c522354605ff291.bindPopup(popup_8cff786d93b66cad6767312bfc91868e)
        ;




            var marker_8d3188991816a7ef071711629cf4faeb = L.marker(
                [12.020000457763672, -5.679999828338623],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_ef2f73543b9b5246f332dcfbbbc0600b = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_523718d3f66eaadaaa655d050ccd12d4 = $(`&lt;div id=&quot;html_523718d3f66eaadaaa655d050ccd12d4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOUORO 1&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -5.679999828338623&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.020000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_ef2f73543b9b5246f332dcfbbbc0600b.setContent(html_523718d3f66eaadaaa655d050ccd12d4);



        marker_8d3188991816a7ef071711629cf4faeb.bindPopup(popup_ef2f73543b9b5246f332dcfbbbc0600b)
        ;




            var marker_395ea66b981f3f24751b84433f84d555 = L.marker(
                [11.680000305175781, -5.579999923706055],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_a3ace0a666db84b5173439aaac807f69 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_6d397233979e2b9ded735c5b66366d7b = $(`&lt;div id=&quot;html_6d397233979e2b9ded735c5b66366d7b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KLELA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -5.579999923706055&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.680000305175781&lt;br&gt;&lt;/div&gt;`)[0];
                popup_a3ace0a666db84b5173439aaac807f69.setContent(html_6d397233979e2b9ded735c5b66366d7b);



        marker_395ea66b981f3f24751b84433f84d555.bindPopup(popup_a3ace0a666db84b5173439aaac807f69)
        ;




            var marker_ac473b870f52edd75f551498b81362c0 = L.marker(
                [13.949999809265137, -5.369999885559082],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_dd93618ec9a1791281c46b6f1066c23f = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_c82cfd400879b02fa0f1ecd89dd54ffd = $(`&lt;div id=&quot;html_c82cfd400879b02fa0f1ecd89dd54ffd&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KE-MACINA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 88.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -5.369999885559082&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.949999809265137&lt;br&gt;&lt;/div&gt;`)[0];
                popup_dd93618ec9a1791281c46b6f1066c23f.setContent(html_c82cfd400879b02fa0f1ecd89dd54ffd);



        marker_ac473b870f52edd75f551498b81362c0.bindPopup(popup_dd93618ec9a1791281c46b6f1066c23f)
        ;




            var marker_67d71905171da3fcdbd6574744a0643f = L.marker(
                [14.170000076293945, -5.019999980926514],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_6df6b23006595e483bf5bf7e95e611b2 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_77c00b7d55086e5618b6fcf0c413c434 = $(`&lt;div id=&quot;html_77c00b7d55086e5618b6fcf0c413c434&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KARA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -5.019999980926514&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.170000076293945&lt;br&gt;&lt;/div&gt;`)[0];
                popup_6df6b23006595e483bf5bf7e95e611b2.setContent(html_77c00b7d55086e5618b6fcf0c413c434);



        marker_67d71905171da3fcdbd6574744a0643f.bindPopup(popup_6df6b23006595e483bf5bf7e95e611b2)
        ;




            var marker_c28b49f00200c99b0465c609faf847d4 = L.marker(
                [13.380000114440918, -4.920000076293945],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_fee26b6406a07daf9b59bcc4d1ed2340 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_3d974486f3ca4baa78fe9deb031ea845 = $(`&lt;div id=&quot;html_3d974486f3ca4baa78fe9deb031ea845&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: BENENY-KEGNY&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -4.920000076293945&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.380000114440918&lt;br&gt;&lt;/div&gt;`)[0];
                popup_fee26b6406a07daf9b59bcc4d1ed2340.setContent(html_3d974486f3ca4baa78fe9deb031ea845);



        marker_c28b49f00200c99b0465c609faf847d4.bindPopup(popup_fee26b6406a07daf9b59bcc4d1ed2340)
        ;




            var marker_8ae74cb961c9f4080970d55d86980826 = L.marker(
                [14.149999618530273, -4.980000019073486],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_138af61ce73069489116617e54d57448 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_ab034663dd3a532c2cac02dc9e646c5f = $(`&lt;div id=&quot;html_ab034663dd3a532c2cac02dc9e646c5f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TILEMBEYA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -4.980000019073486&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.149999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_138af61ce73069489116617e54d57448.setContent(html_ab034663dd3a532c2cac02dc9e646c5f);



        marker_8ae74cb961c9f4080970d55d86980826.bindPopup(popup_138af61ce73069489116617e54d57448)
        ;




            var marker_c495b68c36d534ae05c60cda98a0349a = L.marker(
                [14.020000457763672, -4.25],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_7bb40c5757074f4754112021213a146e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_b742ba07d572ca0d2d6b544e3691812d = $(`&lt;div id=&quot;html_b742ba07d572ca0d2d6b544e3691812d&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: SOFARA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 89.6&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -4.25&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.020000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_7bb40c5757074f4754112021213a146e.setContent(html_b742ba07d572ca0d2d6b544e3691812d);



        marker_c495b68c36d534ae05c60cda98a0349a.bindPopup(popup_7bb40c5757074f4754112021213a146e)
        ;




            var marker_721545c07e0413e03c6e8d461c80df89 = L.marker(
                [14.529999732971191, -4.21999979019165],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_a6f49ea763e457d066d2abb9e2b7d2b2 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_9c59070738b6a24251a1ae52a356a3e7 = $(`&lt;div id=&quot;html_9c59070738b6a24251a1ae52a356a3e7&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: NANTAKA (MOPTI)&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 52.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -4.21999979019165&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.529999732971191&lt;br&gt;&lt;/div&gt;`)[0];
                popup_a6f49ea763e457d066d2abb9e2b7d2b2.setContent(html_9c59070738b6a24251a1ae52a356a3e7);



        marker_721545c07e0413e03c6e8d461c80df89.bindPopup(popup_a6f49ea763e457d066d2abb9e2b7d2b2)
        ;




            var marker_08224e453765c52216a2e575ec6d1ebe = L.marker(
                [14.529999732971191, -4.21999979019165],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_46c9f62d1898c742d747ac7d98338281 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_ebc33c173d18c3bb488bf12c034abcdc = $(`&lt;div id=&quot;html_ebc33c173d18c3bb488bf12c034abcdc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MOPTI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -4.21999979019165&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.529999732971191&lt;br&gt;&lt;/div&gt;`)[0];
                popup_46c9f62d1898c742d747ac7d98338281.setContent(html_ebc33c173d18c3bb488bf12c034abcdc);



        marker_08224e453765c52216a2e575ec6d1ebe.bindPopup(popup_46c9f62d1898c742d747ac7d98338281)
        ;




            var marker_f621c4de9bd7ef02bab94ab53719c37a = L.marker(
                [15.399999618530273, -4.230000019073486],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_7410f0d40a2a3e38e0a783041caee33c = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_2bc61d5e24de8cabf4e81fce8af256d6 = $(`&lt;div id=&quot;html_2bc61d5e24de8cabf4e81fce8af256d6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: AKKA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 94.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -4.230000019073486&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 15.399999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_7410f0d40a2a3e38e0a783041caee33c.setContent(html_2bc61d5e24de8cabf4e81fce8af256d6);



        marker_f621c4de9bd7ef02bab94ab53719c37a.bindPopup(popup_7410f0d40a2a3e38e0a783041caee33c)
        ;




            var marker_21865a1f75e738e1ca610232d76e8e63 = L.marker(
                [16.420000076293945, -3.6500000953674316],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_674d037b15af652410290529c42b97d9 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_aaee6a1299066676d095e95b7c624cb7 = $(`&lt;div id=&quot;html_aaee6a1299066676d095e95b7c624cb7&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GOUNDAM&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -3.6500000953674316&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 16.420000076293945&lt;br&gt;&lt;/div&gt;`)[0];
                popup_674d037b15af652410290529c42b97d9.setContent(html_aaee6a1299066676d095e95b7c624cb7);



        marker_21865a1f75e738e1ca610232d76e8e63.bindPopup(popup_674d037b15af652410290529c42b97d9)
        ;




            var marker_2221e8917dc52dd31465f7430fd46045 = L.marker(
                [16.1299991607666, -3.75],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_274055b62f82d6d0ff343556aace91b5 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_3bc25dd0d492ae0011f2d120e91eaed6 = $(`&lt;div id=&quot;html_3bc25dd0d492ae0011f2d120e91eaed6&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TONKA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -3.75&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 16.1299991607666&lt;br&gt;&lt;/div&gt;`)[0];
                popup_274055b62f82d6d0ff343556aace91b5.setContent(html_3bc25dd0d492ae0011f2d120e91eaed6);



        marker_2221e8917dc52dd31465f7430fd46045.bindPopup(popup_274055b62f82d6d0ff343556aace91b5)
        ;




            var marker_544350e4dad4afe6efb5c19ff96183da = L.marker(
                [15.819999694824219, -3.700000047683716],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_7e08a0f3d8674058fef83f70b4dba7e0 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_2784f07da11f8cd9b4ba6ad884a92c53 = $(`&lt;div id=&quot;html_2784f07da11f8cd9b4ba6ad884a92c53&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: SARAFERE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -3.700000047683716&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 15.819999694824219&lt;br&gt;&lt;/div&gt;`)[0];
                popup_7e08a0f3d8674058fef83f70b4dba7e0.setContent(html_2784f07da11f8cd9b4ba6ad884a92c53);



        marker_544350e4dad4afe6efb5c19ff96183da.bindPopup(popup_7e08a0f3d8674058fef83f70b4dba7e0)
        ;




            var marker_fd22d33d268f5d50674bd6abcde3da3b = L.marker(
                [16.266700744628906, -3.3833000659942627],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_4d87d196b26beee83f854cb8c16c5529 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_7a9f2a7a42070b348d4477ff34cfc7c8 = $(`&lt;div id=&quot;html_7a9f2a7a42070b348d4477ff34cfc7c8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DIRE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 47.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -3.3833000659942627&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 16.266700744628906&lt;br&gt;&lt;/div&gt;`)[0];
                popup_4d87d196b26beee83f854cb8c16c5529.setContent(html_7a9f2a7a42070b348d4477ff34cfc7c8);



        marker_fd22d33d268f5d50674bd6abcde3da3b.bindPopup(popup_4d87d196b26beee83f854cb8c16c5529)
        ;




            var marker_d6e612b2fc5c5bceed9f4b491adfda93 = L.marker(
                [16.66659927368164, -3.0332999229431152],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_606659ef3b1261652b950843338d07f0 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_887276565787fddde8c3877dcd2e822a = $(`&lt;div id=&quot;html_887276565787fddde8c3877dcd2e822a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KORYOUME&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 47.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -3.0332999229431152&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 16.66659927368164&lt;br&gt;&lt;/div&gt;`)[0];
                popup_606659ef3b1261652b950843338d07f0.setContent(html_887276565787fddde8c3877dcd2e822a);



        marker_d6e612b2fc5c5bceed9f4b491adfda93.bindPopup(popup_606659ef3b1261652b950843338d07f0)
        ;




            var marker_cd7a7dbe094c7b2b22130efe36d78a14 = L.marker(
                [16.93000030517578, -0.5799999833106995],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_f2c9a0f7c88bd29fd8eea000f19f5f74 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_f046851de190b290c1e6b5dc0d4edb60 = $(`&lt;div id=&quot;html_f046851de190b290c1e6b5dc0d4edb60&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TOSSAYE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -0.5799999833106995&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 16.93000030517578&lt;br&gt;&lt;/div&gt;`)[0];
                popup_f2c9a0f7c88bd29fd8eea000f19f5f74.setContent(html_f046851de190b290c1e6b5dc0d4edb60);



        marker_cd7a7dbe094c7b2b22130efe36d78a14.bindPopup(popup_f2c9a0f7c88bd29fd8eea000f19f5f74)
        ;




            var marker_620d5d9cd53f5465a65aa88667b014f0 = L.marker(
                [15.66670036315918, 0.5],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_c390d2f8f7eaee68881b0fa5e9be7f1e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_cc71e4eaa6bcb0428dfc7e112cc81428 = $(`&lt;div id=&quot;html_cc71e4eaa6bcb0428dfc7e112cc81428&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: ANSONGO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 56.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 0.5&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 15.66670036315918&lt;br&gt;&lt;/div&gt;`)[0];
                popup_c390d2f8f7eaee68881b0fa5e9be7f1e.setContent(html_cc71e4eaa6bcb0428dfc7e112cc81428);



        marker_620d5d9cd53f5465a65aa88667b014f0.bindPopup(popup_c390d2f8f7eaee68881b0fa5e9be7f1e)
        ;




            var marker_e462462609cf7c4bda0cd76146c4bac3 = L.marker(
                [14.619999885559082, 0.30000001192092896],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_559e2b3443ddbf785982ddbc41655780 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_1ddf185b33acd25f8b42df184cd6256b = $(`&lt;div id=&quot;html_1ddf185b33acd25f8b42df184cd6256b&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DOLBEL&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 86.1&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 0.30000001192092896&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.619999885559082&lt;br&gt;&lt;/div&gt;`)[0];
                popup_559e2b3443ddbf785982ddbc41655780.setContent(html_1ddf185b33acd25f8b42df184cd6256b);



        marker_e462462609cf7c4bda0cd76146c4bac3.bindPopup(popup_559e2b3443ddbf785982ddbc41655780)
        ;




            var marker_2a9cb4bb035d389b7fb9ce587e60a3ab = L.marker(
                [14.75, 0.6000000238418579],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_cad741e1f24ad966317a7c552857a0ff = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_94ab2cf4770c6408ded403f9026ef948 = $(`&lt;div id=&quot;html_94ab2cf4770c6408ded403f9026ef948&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: ALCONGUI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 89.8&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 0.6000000238418579&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.75&lt;br&gt;&lt;/div&gt;`)[0];
                popup_cad741e1f24ad966317a7c552857a0ff.setContent(html_94ab2cf4770c6408ded403f9026ef948);



        marker_2a9cb4bb035d389b7fb9ce587e60a3ab.bindPopup(popup_cad741e1f24ad966317a7c552857a0ff)
        ;




            var marker_057783c4ee815d50e74b05137ddd6335 = L.marker(
                [14.616700172424316, 0.983299970626831],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_478c7e857c2e2fdfdd767caa0b13cc2e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_5a587d3b23c34ffa7e07de337105e0f7 = $(`&lt;div id=&quot;html_5a587d3b23c34ffa7e07de337105e0f7&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KANDADJI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 26.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 0.983299970626831&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.616700172424316&lt;br&gt;&lt;/div&gt;`)[0];
                popup_478c7e857c2e2fdfdd767caa0b13cc2e.setContent(html_5a587d3b23c34ffa7e07de337105e0f7);



        marker_057783c4ee815d50e74b05137ddd6335.bindPopup(popup_478c7e857c2e2fdfdd767caa0b13cc2e)
        ;




            var marker_55ae5a2646190b75c4a449783e6dfb65 = L.marker(
                [14.020000457763672, 0.75],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_42e75e562409677aa530364208869cd9 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_51598b483dfff65f92827bced0de6d5e = $(`&lt;div id=&quot;html_51598b483dfff65f92827bced0de6d5e&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TERA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 94.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 0.75&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.020000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_42e75e562409677aa530364208869cd9.setContent(html_51598b483dfff65f92827bced0de6d5e);



        marker_55ae5a2646190b75c4a449783e6dfb65.bindPopup(popup_42e75e562409677aa530364208869cd9)
        ;




            var marker_41e3489144b17a1f23d0672be82c794e = L.marker(
                [13.850000381469727, 1.4700000286102295],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_2e1b674f3788f435e31c133295f8c6fe = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_76bf412d1eb773820ae834e1f968fe08 = $(`&lt;div id=&quot;html_76bf412d1eb773820ae834e1f968fe08&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KAKASSI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 86.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 1.4700000286102295&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.850000381469727&lt;br&gt;&lt;/div&gt;`)[0];
                popup_2e1b674f3788f435e31c133295f8c6fe.setContent(html_76bf412d1eb773820ae834e1f968fe08);



        marker_41e3489144b17a1f23d0672be82c794e.bindPopup(popup_2e1b674f3788f435e31c133295f8c6fe)
        ;




            var marker_ab69602e14558ba3a93b54e76da1b44f = L.marker(
                [13.73330020904541, 1.6167000532150269],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_65755d90e00f30ebaa9a562aa319115e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_4d0f5124824e21a7848b3d688f237c2f = $(`&lt;div id=&quot;html_4d0f5124824e21a7848b3d688f237c2f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GARBE-KOUROU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 41.6&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 1.6167000532150269&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.73330020904541&lt;br&gt;&lt;/div&gt;`)[0];
                popup_65755d90e00f30ebaa9a562aa319115e.setContent(html_4d0f5124824e21a7848b3d688f237c2f);



        marker_ab69602e14558ba3a93b54e76da1b44f.bindPopup(popup_65755d90e00f30ebaa9a562aa319115e)
        ;




            var marker_5af4e79b59672067a6e2ba8257eb27d0 = L.marker(
                [13.520000457763672, 2.0899999141693115],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_672a66ab77e6313941372c28eed331b6 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_0aee587f997f4751a474a4e36d70484f = $(`&lt;div id=&quot;html_0aee587f997f4751a474a4e36d70484f&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: NIAMEY&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 0.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.0899999141693115&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.520000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_672a66ab77e6313941372c28eed331b6.setContent(html_0aee587f997f4751a474a4e36d70484f);



        marker_5af4e79b59672067a6e2ba8257eb27d0.bindPopup(popup_672a66ab77e6313941372c28eed331b6)
        ;




            var marker_77469a20ebf8d44b999098a2924b02d9 = L.marker(
                [12.90625, 2.314579963684082],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_7f128d3c77aae2de11a38e7244344247 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_c45d8b921b30077528b7818a067f45ea = $(`&lt;div id=&quot;html_c45d8b921b30077528b7818a067f45ea&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DIONGORE AMONT&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 91.5&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.314579963684082&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.90625&lt;br&gt;&lt;/div&gt;`)[0];
                popup_7f128d3c77aae2de11a38e7244344247.setContent(html_c45d8b921b30077528b7818a067f45ea);



        marker_77469a20ebf8d44b999098a2924b02d9.bindPopup(popup_7f128d3c77aae2de11a38e7244344247)
        ;




            var marker_26a5a14091e627dae33dcf87dfcc3b7d = L.marker(
                [12.739999771118164, 2.240000009536743],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_ed15f1a13676b492b65864c696c84512 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_2e6447787348c31b3d9db5330c4b92ae = $(`&lt;div id=&quot;html_2e6447787348c31b3d9db5330c4b92ae&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TAMOU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 98.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.240000009536743&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.739999771118164&lt;br&gt;&lt;/div&gt;`)[0];
                popup_ed15f1a13676b492b65864c696c84512.setContent(html_2e6447787348c31b3d9db5330c4b92ae);



        marker_26a5a14091e627dae33dcf87dfcc3b7d.bindPopup(popup_ed15f1a13676b492b65864c696c84512)
        ;




            var marker_c94bfef4e681087a28d50cf260f709ea = L.marker(
                [12.470000267028809, 2.4200000762939453],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_09fb820c1cf8c60ade82d5be23c30453 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_067b929a7b5af9560127c139e9a1fade = $(`&lt;div id=&quot;html_067b929a7b5af9560127c139e9a1fade&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: CAMPEMENT DU DOUBLE VE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 88.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.4200000762939453&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.470000267028809&lt;br&gt;&lt;/div&gt;`)[0];
                popup_09fb820c1cf8c60ade82d5be23c30453.setContent(html_067b929a7b5af9560127c139e9a1fade);



        marker_c94bfef4e681087a28d50cf260f709ea.bindPopup(popup_09fb820c1cf8c60ade82d5be23c30453)
        ;




            var marker_4f19cdcbbd24bcc711505361ff7d5a9c = L.marker(
                [12.58329963684082, 2.6166999340057373],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_10f06c64cce356b5e3948b54fc0fdd4b = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_e380c0d72ec034930b99911d16991d75 = $(`&lt;div id=&quot;html_e380c0d72ec034930b99911d16991d75&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: W&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 68.6&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.6166999340057373&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.58329963684082&lt;br&gt;&lt;/div&gt;`)[0];
                popup_10f06c64cce356b5e3948b54fc0fdd4b.setContent(html_e380c0d72ec034930b99911d16991d75);



        marker_4f19cdcbbd24bcc711505361ff7d5a9c.bindPopup(popup_10f06c64cce356b5e3948b54fc0fdd4b)
        ;




            var marker_4381cccd10f109aed64c43d1638fcbd4 = L.marker(
                [13.880000114440918, 5.329999923706055],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_a9f955fcc9e5725fa801dee07c8e8d45 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_691cfe62a6c7d4dd7ba2b9df6f652330 = $(`&lt;div id=&quot;html_691cfe62a6c7d4dd7ba2b9df6f652330&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TSERNAOUA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 86.8&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 5.329999923706055&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.880000114440918&lt;br&gt;&lt;/div&gt;`)[0];
                popup_a9f955fcc9e5725fa801dee07c8e8d45.setContent(html_691cfe62a6c7d4dd7ba2b9df6f652330);



        marker_4381cccd10f109aed64c43d1638fcbd4.bindPopup(popup_a9f955fcc9e5725fa801dee07c8e8d45)
        ;




            var marker_6414524497d719eb70abee71169e5afd = L.marker(
                [14.5, 5.369999885559082],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_64f0a0ac1186f251eeecff8af0b4bbcc = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_1d0ac93633a2749d295f788066004aef = $(`&lt;div id=&quot;html_1d0ac93633a2749d295f788066004aef&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: BADEGUICHERI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 5.369999885559082&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.5&lt;br&gt;&lt;/div&gt;`)[0];
                popup_64f0a0ac1186f251eeecff8af0b4bbcc.setContent(html_1d0ac93633a2749d295f788066004aef);



        marker_6414524497d719eb70abee71169e5afd.bindPopup(popup_64f0a0ac1186f251eeecff8af0b4bbcc)
        ;




            var marker_98bc6d012a4c9aecd520d0a4cb6fd61d = L.marker(
                [13.670000076293945, 6.769999980926514],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_abd5aecac2302e930db0e82784174e7c = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_6e0ac46df1dc6132f3a5580d0eb033ec = $(`&lt;div id=&quot;html_6e0ac46df1dc6132f3a5580d0eb033ec&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GUINDAM ROUMDJI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 6.769999980926514&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.670000076293945&lt;br&gt;&lt;/div&gt;`)[0];
                popup_abd5aecac2302e930db0e82784174e7c.setContent(html_6e0ac46df1dc6132f3a5580d0eb033ec);



        marker_98bc6d012a4c9aecd520d0a4cb6fd61d.bindPopup(popup_abd5aecac2302e930db0e82784174e7c)
        ;




            var marker_8acb51789a392192ffa041c60c1d5ae0 = L.marker(
                [13.319999694824219, 7.170000076293945],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_8bf0493c7c605348b2ef7aee852f94ef = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_d95196aad2dcf68488ad60ee67f100ff = $(`&lt;div id=&quot;html_d95196aad2dcf68488ad60ee67f100ff&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MADAROUNFA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 7.170000076293945&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.319999694824219&lt;br&gt;&lt;/div&gt;`)[0];
                popup_8bf0493c7c605348b2ef7aee852f94ef.setContent(html_d95196aad2dcf68488ad60ee67f100ff);



        marker_8acb51789a392192ffa041c60c1d5ae0.bindPopup(popup_8bf0493c7c605348b2ef7aee852f94ef)
        ;




            var marker_2feaca17f67c663e79679fe2e21eb95a = L.marker(
                [13.149999618530273, 7.21999979019165],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_b2133f36e78d77a5fe31f1cd79bf5a98 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_8d0bd6474fada4b7dfd776d25133b2de = $(`&lt;div id=&quot;html_8d0bd6474fada4b7dfd776d25133b2de&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: NIELLOUA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 88.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 7.21999979019165&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 13.149999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_b2133f36e78d77a5fe31f1cd79bf5a98.setContent(html_8d0bd6474fada4b7dfd776d25133b2de);



        marker_2feaca17f67c663e79679fe2e21eb95a.bindPopup(popup_b2133f36e78d77a5fe31f1cd79bf5a98)
        ;




            var marker_8f31c30b9bae1e5a59558c80279aa949 = L.marker(
                [9.699999809265137, -7.789999961853027],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_fc54572ac28d7783a564ba8c0bb5fac1 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_7f953c7ee760b19e86527893146a52c5 = $(`&lt;div id=&quot;html_7f953c7ee760b19e86527893146a52c5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: IRADOUGOU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.789999961853027&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.699999809265137&lt;br&gt;&lt;/div&gt;`)[0];
                popup_fc54572ac28d7783a564ba8c0bb5fac1.setContent(html_7f953c7ee760b19e86527893146a52c5);



        marker_8f31c30b9bae1e5a59558c80279aa949.bindPopup(popup_fc54572ac28d7783a564ba8c0bb5fac1)
        ;




            var marker_1491f813ca33b1558fbec8785cb46293 = L.marker(
                [10.0600004196167, -7.579999923706055],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_c1198279064e46bf3effd743d280a15a = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_21764deff53a23687992ac90f2918871 = $(`&lt;div id=&quot;html_21764deff53a23687992ac90f2918871&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DJIRILA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.579999923706055&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.0600004196167&lt;br&gt;&lt;/div&gt;`)[0];
                popup_c1198279064e46bf3effd743d280a15a.setContent(html_21764deff53a23687992ac90f2918871);



        marker_1491f813ca33b1558fbec8785cb46293.bindPopup(popup_c1198279064e46bf3effd743d280a15a)
        ;




            var marker_e2ab17a7a509289c16c50d1bceebd80f = L.marker(
                [9.84000015258789, -7.570000171661377],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_421cccf58c6c8b12186eae8a434aa9ac = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_4162f8521d95a02f6216103c47714695 = $(`&lt;div id=&quot;html_4162f8521d95a02f6216103c47714695&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: SAMATIGUILA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.570000171661377&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.84000015258789&lt;br&gt;&lt;/div&gt;`)[0];
                popup_421cccf58c6c8b12186eae8a434aa9ac.setContent(html_4162f8521d95a02f6216103c47714695);



        marker_e2ab17a7a509289c16c50d1bceebd80f.bindPopup(popup_421cccf58c6c8b12186eae8a434aa9ac)
        ;




            var marker_69464bb8c6af69cde032de04b3431d26 = L.marker(
                [9.90999984741211, -7.420000076293945],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_581e37c6aab75ba2b461c4c7b19e8e52 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_9a066b3332f27cb9bf6d283de66aeece = $(`&lt;div id=&quot;html_9a066b3332f27cb9bf6d283de66aeece&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: ZIEMOUGOULA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.420000076293945&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.90999984741211&lt;br&gt;&lt;/div&gt;`)[0];
                popup_581e37c6aab75ba2b461c4c7b19e8e52.setContent(html_9a066b3332f27cb9bf6d283de66aeece);



        marker_69464bb8c6af69cde032de04b3431d26.bindPopup(popup_581e37c6aab75ba2b461c4c7b19e8e52)
        ;




            var marker_6906584688f282e3344cc0e3c085944c = L.marker(
                [10.4399995803833, -7.449999809265137],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_59dab5d49b18df84b3f848467a40944e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_f65d9f677f78a849d0b1ebe0108a2321 = $(`&lt;div id=&quot;html_f65d9f677f78a849d0b1ebe0108a2321&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MANANKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -7.449999809265137&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.4399995803833&lt;br&gt;&lt;/div&gt;`)[0];
                popup_59dab5d49b18df84b3f848467a40944e.setContent(html_f65d9f677f78a849d0b1ebe0108a2321);



        marker_6906584688f282e3344cc0e3c085944c.bindPopup(popup_59dab5d49b18df84b3f848467a40944e)
        ;




            var marker_e3314239a40f70fd5a16e85b0952224b = L.marker(
                [10.09000015258789, -6.909999847412109],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_ee4de1f6cb7f7d30d58145fbb8422df9 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_54a8932c01c064309e6a930fc5b327d1 = $(`&lt;div id=&quot;html_54a8932c01c064309e6a930fc5b327d1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: WAHIRE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.7&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.909999847412109&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.09000015258789&lt;br&gt;&lt;/div&gt;`)[0];
                popup_ee4de1f6cb7f7d30d58145fbb8422df9.setContent(html_54a8932c01c064309e6a930fc5b327d1);



        marker_e3314239a40f70fd5a16e85b0952224b.bindPopup(popup_ee4de1f6cb7f7d30d58145fbb8422df9)
        ;




            var marker_eb7105a2c16ffbc6c7262d9af1aeae8d = L.marker(
                [10.09000015258789, -6.929999828338623],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_f059235e5a365d36108e0124e8bc9a46 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_a57ed300ce9c59dc83cc8a8208f00860 = $(`&lt;div id=&quot;html_a57ed300ce9c59dc83cc8a8208f00860&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: WAHIRE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.929999828338623&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.09000015258789&lt;br&gt;&lt;/div&gt;`)[0];
                popup_f059235e5a365d36108e0124e8bc9a46.setContent(html_a57ed300ce9c59dc83cc8a8208f00860);



        marker_eb7105a2c16ffbc6c7262d9af1aeae8d.bindPopup(popup_f059235e5a365d36108e0124e8bc9a46)
        ;




            var marker_a4cceb4f38405935fae9f602242338a2 = L.marker(
                [10.631041526794434, -6.657205104827881],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_9ed2abe5ad9e1339fa7725c6845a0d25 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_ab39c719c72152e4eda4bbadd626f6d8 = $(`&lt;div id=&quot;html_ab39c719c72152e4eda4bbadd626f6d8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DEBETE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 94.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.657205104827881&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.631041526794434&lt;br&gt;&lt;/div&gt;`)[0];
                popup_9ed2abe5ad9e1339fa7725c6845a0d25.setContent(html_ab39c719c72152e4eda4bbadd626f6d8);



        marker_a4cceb4f38405935fae9f602242338a2.bindPopup(popup_9ed2abe5ad9e1339fa7725c6845a0d25)
        ;




            var marker_891df9623ad0383a08bd589f0eec8c6c = L.marker(
                [9.522919654846191, -6.606249809265137],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_2bf1957719be8b934835b10fffb5ac81 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_0788701d6ce509eae9a44f9d7f87fcb8 = $(`&lt;div id=&quot;html_0788701d6ce509eae9a44f9d7f87fcb8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GUINGUERINI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.606249809265137&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.522919654846191&lt;br&gt;&lt;/div&gt;`)[0];
                popup_2bf1957719be8b934835b10fffb5ac81.setContent(html_0788701d6ce509eae9a44f9d7f87fcb8);



        marker_891df9623ad0383a08bd589f0eec8c6c.bindPopup(popup_2bf1957719be8b934835b10fffb5ac81)
        ;




            var marker_6dcdc06b03e078b838b18c981616d9f8 = L.marker(
                [9.510000228881836, -6.360000133514404],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_2adba387d652db9973fefe239e535420 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_b6716a2998e00ffac5fd936a7cedbc45 = $(`&lt;div id=&quot;html_b6716a2998e00ffac5fd936a7cedbc45&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: PONONDOUGOU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 98.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.360000133514404&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.510000228881836&lt;br&gt;&lt;/div&gt;`)[0];
                popup_2adba387d652db9973fefe239e535420.setContent(html_b6716a2998e00ffac5fd936a7cedbc45);



        marker_6dcdc06b03e078b838b18c981616d9f8.bindPopup(popup_2adba387d652db9973fefe239e535420)
        ;




            var marker_c49179717dd5472529aef1277a7ebd6d = L.marker(
                [9.581250190734863, -6.510419845581055],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_d7fe2399674c5e6a8ff6e15d87cdc931 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_038a9462a0ab0190e1a9456aefdd77e3 = $(`&lt;div id=&quot;html_038a9462a0ab0190e1a9456aefdd77e3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TOMBOUGOU 1&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.510419845581055&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.581250190734863&lt;br&gt;&lt;/div&gt;`)[0];
                popup_d7fe2399674c5e6a8ff6e15d87cdc931.setContent(html_038a9462a0ab0190e1a9456aefdd77e3);



        marker_c49179717dd5472529aef1277a7ebd6d.bindPopup(popup_d7fe2399674c5e6a8ff6e15d87cdc931)
        ;




            var marker_d44f48537742164b1c7a31d45ab9925d = L.marker(
                [9.577079772949219, -6.514579772949219],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_4df4dd2cd29f3253a90d1946911f186e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_dc5f58be7684c0674505187572d813f9 = $(`&lt;div id=&quot;html_dc5f58be7684c0674505187572d813f9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TOMBOUGOU 2&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.514579772949219&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.577079772949219&lt;br&gt;&lt;/div&gt;`)[0];
                popup_4df4dd2cd29f3253a90d1946911f186e.setContent(html_dc5f58be7684c0674505187572d813f9);



        marker_d44f48537742164b1c7a31d45ab9925d.bindPopup(popup_4df4dd2cd29f3253a90d1946911f186e)
        ;




            var marker_5eb9137e72d8146d4d8c99fa477c9eb9 = L.marker(
                [9.850000381469727, -6.360000133514404],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_31190f9540ec25e76314af27838b7d14 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_1d1df9f79c4f5de57b6beb97cae259e4 = $(`&lt;div id=&quot;html_1d1df9f79c4f5de57b6beb97cae259e4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOUTO AVAL&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.360000133514404&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.850000381469727&lt;br&gt;&lt;/div&gt;`)[0];
                popup_31190f9540ec25e76314af27838b7d14.setContent(html_1d1df9f79c4f5de57b6beb97cae259e4);



        marker_5eb9137e72d8146d4d8c99fa477c9eb9.bindPopup(popup_31190f9540ec25e76314af27838b7d14)
        ;




            var marker_96166144bfff25e934866911de9759d3 = L.marker(
                [9.883299827575684, -6.366700172424316],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_debb36d336c9aaee015564ac6d638865 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_55afec6472aae1059a1736258f120bcc = $(`&lt;div id=&quot;html_55afec6472aae1059a1736258f120bcc&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOUTO AMONT&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 59.3&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.366700172424316&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.883299827575684&lt;br&gt;&lt;/div&gt;`)[0];
                popup_debb36d336c9aaee015564ac6d638865.setContent(html_55afec6472aae1059a1736258f120bcc);



        marker_96166144bfff25e934866911de9759d3.bindPopup(popup_debb36d336c9aaee015564ac6d638865)
        ;




            var marker_14b33ee2d8b1265972edeca08dbaa9b8 = L.marker(
                [10.630000114440918, -6.21999979019165],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_72401a0004436ff2274a2db33eda5de7 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_e2b0b41098459689a0edd008e95d2f33 = $(`&lt;div id=&quot;html_e2b0b41098459689a0edd008e95d2f33&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: PAPARA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 95.8&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -6.21999979019165&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.630000114440918&lt;br&gt;&lt;/div&gt;`)[0];
                popup_72401a0004436ff2274a2db33eda5de7.setContent(html_e2b0b41098459689a0edd008e95d2f33);



        marker_14b33ee2d8b1265972edeca08dbaa9b8.bindPopup(popup_72401a0004436ff2274a2db33eda5de7)
        ;




            var marker_0a7c8d2805bebe9e97438325f63d55c6 = L.marker(
                [9.6899995803833, 15.5],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_4cd188ab44ebc98237dd96364169157b = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_a52b994aa3143827adc3d4888470a660 = $(`&lt;div id=&quot;html_a52b994aa3143827adc3d4888470a660&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: GOUNOU-GAYA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 87.1&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 15.5&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.6899995803833&lt;br&gt;&lt;/div&gt;`)[0];
                popup_4cd188ab44ebc98237dd96364169157b.setContent(html_a52b994aa3143827adc3d4888470a660);



        marker_0a7c8d2805bebe9e97438325f63d55c6.bindPopup(popup_4cd188ab44ebc98237dd96364169157b)
        ;




            var marker_c6ea878f3fb50814025e178b1662b8be = L.marker(
                [9.279999732971191, 15.5],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_cd6320cfa84b9d33cb836929613adc89 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_eecf55d27a7fc57b0afdb154a8795296 = $(`&lt;div id=&quot;html_eecf55d27a7fc57b0afdb154a8795296&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: PONT CAROL&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 76.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 15.5&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.279999732971191&lt;br&gt;&lt;/div&gt;`)[0];
                popup_cd6320cfa84b9d33cb836929613adc89.setContent(html_eecf55d27a7fc57b0afdb154a8795296);



        marker_c6ea878f3fb50814025e178b1662b8be.bindPopup(popup_cd6320cfa84b9d33cb836929613adc89)
        ;




            var marker_9698fd911d9c54b0e7e6c1e6a9435227 = L.marker(
                [10.706250190734863, -11.102080345153809],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_388fb25a2a2dfd92038417cfead249ed = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_9ca73850a04301fc6e6116acf99b828a = $(`&lt;div id=&quot;html_9ca73850a04301fc6e6116acf99b828a&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DABOLA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -11.102080345153809&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.706250190734863&lt;br&gt;&lt;/div&gt;`)[0];
                popup_388fb25a2a2dfd92038417cfead249ed.setContent(html_9ca73850a04301fc6e6116acf99b828a);



        marker_9698fd911d9c54b0e7e6c1e6a9435227.bindPopup(popup_388fb25a2a2dfd92038417cfead249ed)
        ;




            var marker_377340a257ab0df47290dd153c255a68 = L.marker(
                [10.033300399780273, -10.75],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_52706438c2889a1451c6740b6f3d52d1 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_cd752795e55a4122961484eff58d2227 = $(`&lt;div id=&quot;html_cd752795e55a4122961484eff58d2227&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: FARANAH&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 49.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -10.75&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.033300399780273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_52706438c2889a1451c6740b6f3d52d1.setContent(html_cd752795e55a4122961484eff58d2227);



        marker_377340a257ab0df47290dd153c255a68.bindPopup(popup_52706438c2889a1451c6740b6f3d52d1)
        ;




            var marker_c3afeb9b970e5408db2e7f11aa66b3ad = L.marker(
                [9.180000305175781, -10.0600004196167],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_c6e4ed070c4e957fffdc113ae121c6a3 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_87700554ad9e22b7e8de818d0dfea4d9 = $(`&lt;div id=&quot;html_87700554ad9e22b7e8de818d0dfea4d9&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KISSIDOUGOU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -10.0600004196167&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.180000305175781&lt;br&gt;&lt;/div&gt;`)[0];
                popup_c6e4ed070c4e957fffdc113ae121c6a3.setContent(html_87700554ad9e22b7e8de818d0dfea4d9);



        marker_c3afeb9b970e5408db2e7f11aa66b3ad.bindPopup(popup_c6e4ed070c4e957fffdc113ae121c6a3)
        ;




            var marker_53133bc9a7ef0dbd73c9bd33e73fba06 = L.marker(
                [10.649999618530273, -9.866700172424316],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_d459c4b1956f528b948e69525a6eec9b = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_bc10d418813ebb7e893764acf3bb2d69 = $(`&lt;div id=&quot;html_bc10d418813ebb7e893764acf3bb2d69&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOUROUSSA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 51.5&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.866700172424316&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.649999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_d459c4b1956f528b948e69525a6eec9b.setContent(html_bc10d418813ebb7e893764acf3bb2d69);



        marker_53133bc9a7ef0dbd73c9bd33e73fba06.bindPopup(popup_d459c4b1956f528b948e69525a6eec9b)
        ;




            var marker_7a3c4dcf93b2ce2999d245bd37a44829 = L.marker(
                [10.600000381469727, -9.729999542236328],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_1d77ce03d2dddc17f1b9b82e489b3a3e = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_6dd40d8fd35a9c5b357829a99115aa72 = $(`&lt;div id=&quot;html_6dd40d8fd35a9c5b357829a99115aa72&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: BARO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.729999542236328&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.600000381469727&lt;br&gt;&lt;/div&gt;`)[0];
                popup_1d77ce03d2dddc17f1b9b82e489b3a3e.setContent(html_6dd40d8fd35a9c5b357829a99115aa72);



        marker_7a3c4dcf93b2ce2999d245bd37a44829.bindPopup(popup_1d77ce03d2dddc17f1b9b82e489b3a3e)
        ;




            var marker_b82caa7953cc2b3de6354dbb9d312d37 = L.marker(
                [10.399999618530273, -9.800000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_2e9f693b44910e3a1c8d796d02fad8f1 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_445e156fde70b1c28597eb4eefebd254 = $(`&lt;div id=&quot;html_445e156fde70b1c28597eb4eefebd254&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MOLOKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.800000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.399999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_2e9f693b44910e3a1c8d796d02fad8f1.setContent(html_445e156fde70b1c28597eb4eefebd254);



        marker_b82caa7953cc2b3de6354dbb9d312d37.bindPopup(popup_2e9f693b44910e3a1c8d796d02fad8f1)
        ;




            var marker_eb2bbd6253544618a3511001ccad4e15 = L.marker(
                [10.383299827575684, -9.300000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_4148d9bd4199dd74151e9c7fe6c509da = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_e08ac6d820e9ffef738c29e8f88d8593 = $(`&lt;div id=&quot;html_e08ac6d820e9ffef738c29e8f88d8593&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KANKAN&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 7.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.300000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.383299827575684&lt;br&gt;&lt;/div&gt;`)[0];
                popup_4148d9bd4199dd74151e9c7fe6c509da.setContent(html_e08ac6d820e9ffef738c29e8f88d8593);



        marker_eb2bbd6253544618a3511001ccad4e15.bindPopup(popup_4148d9bd4199dd74151e9c7fe6c509da)
        ;




            var marker_c8fc64d6312a0ee6810b7ff12872be52 = L.marker(
                [9.029999732971191, -9.0],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_1c3a8bc8d14bc449fca7b95e72c0f34d = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_a79505ba8a03a489c55f5e90e6ec1932 = $(`&lt;div id=&quot;html_a79505ba8a03a489c55f5e90e6ec1932&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KONSANKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.0&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.029999732971191&lt;br&gt;&lt;/div&gt;`)[0];
                popup_1c3a8bc8d14bc449fca7b95e72c0f34d.setContent(html_a79505ba8a03a489c55f5e90e6ec1932);



        marker_c8fc64d6312a0ee6810b7ff12872be52.bindPopup(popup_1c3a8bc8d14bc449fca7b95e72c0f34d)
        ;




            var marker_f5e6292d87737946b8423af25be57606 = L.marker(
                [9.270000457763672, -9.020000457763672],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_2abff4d821bace8f7d04f67e6dd02b65 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_bc51aff303fbb5043392490dad8bc276 = $(`&lt;div id=&quot;html_bc51aff303fbb5043392490dad8bc276&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KEROUANE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 48.5&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.020000457763672&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 9.270000457763672&lt;br&gt;&lt;/div&gt;`)[0];
                popup_2abff4d821bace8f7d04f67e6dd02b65.setContent(html_bc51aff303fbb5043392490dad8bc276);



        marker_f5e6292d87737946b8423af25be57606.bindPopup(popup_2abff4d821bace8f7d04f67e6dd02b65)
        ;




            var marker_574cdbf73345a46d5064398b7f42a4c3 = L.marker(
                [11.369999885559082, -9.609999656677246],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_1875d5f60dfb99ee427dd92b7a565492 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_95b3a629c337be4d290d6e78d20cd490 = $(`&lt;div id=&quot;html_95b3a629c337be4d290d6e78d20cd490&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: OUARAN&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.609999656677246&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.369999885559082&lt;br&gt;&lt;/div&gt;`)[0];
                popup_1875d5f60dfb99ee427dd92b7a565492.setContent(html_95b3a629c337be4d290d6e78d20cd490);



        marker_574cdbf73345a46d5064398b7f42a4c3.bindPopup(popup_1875d5f60dfb99ee427dd92b7a565492)
        ;




            var marker_7d2499ae1b019729eef130b071ce6351 = L.marker(
                [11.25, -10.616666793823242],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_db745ec27230d4a2ec67fedacdfbf314 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_80628753d6c055da3600ca49da02be70 = $(`&lt;div id=&quot;html_80628753d6c055da3600ca49da02be70&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TINKISSO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -10.616666793823242&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.25&lt;br&gt;&lt;/div&gt;`)[0];
                popup_db745ec27230d4a2ec67fedacdfbf314.setContent(html_80628753d6c055da3600ca49da02be70);



        marker_7d2499ae1b019729eef130b071ce6351.bindPopup(popup_db745ec27230d4a2ec67fedacdfbf314)
        ;




            var marker_4856a448551e8a6bd33501fca387ce4f = L.marker(
                [11.364580154418945, -9.160420417785645],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_d88177e88c69c46b29f8cdb99b657d3a = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_d16876f996f0871f370c7718269881ce = $(`&lt;div id=&quot;html_d16876f996f0871f370c7718269881ce&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: TIGUIBERY&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -9.160420417785645&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.364580154418945&lt;br&gt;&lt;/div&gt;`)[0];
                popup_d88177e88c69c46b29f8cdb99b657d3a.setContent(html_d16876f996f0871f370c7718269881ce);



        marker_4856a448551e8a6bd33501fca387ce4f.bindPopup(popup_d88177e88c69c46b29f8cdb99b657d3a)
        ;




            var marker_7b03f9261485ad37a2036c40fabfcb9a = L.marker(
                [11.420000076293945, -8.90999984741211],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_00e2f08f89b2e1bbbe5b3a3d68ea43ea = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_0b4065c4fb2bff2b42aede80dcd08b27 = $(`&lt;div id=&quot;html_0b4065c4fb2bff2b42aede80dcd08b27&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: DIALAKORO&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.90999984741211&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.420000076293945&lt;br&gt;&lt;/div&gt;`)[0];
                popup_00e2f08f89b2e1bbbe5b3a3d68ea43ea.setContent(html_0b4065c4fb2bff2b42aede80dcd08b27);



        marker_7b03f9261485ad37a2036c40fabfcb9a.bindPopup(popup_00e2f08f89b2e1bbbe5b3a3d68ea43ea)
        ;




            var marker_3335c9a4d5bc1e1ef4858b73f7f495e8 = L.marker(
                [10.630000114440918, -8.680000305175781],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_e84b7a4003b0f653c4d813bda1b52906 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_1a0c670d790140d724a88016542f29d1 = $(`&lt;div id=&quot;html_1a0c670d790140d724a88016542f29d1&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MANDIANA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -8.680000305175781&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.630000114440918&lt;br&gt;&lt;/div&gt;`)[0];
                popup_e84b7a4003b0f653c4d813bda1b52906.setContent(html_1a0c670d790140d724a88016542f29d1);



        marker_3335c9a4d5bc1e1ef4858b73f7f495e8.bindPopup(popup_e84b7a4003b0f653c4d813bda1b52906)
        ;




            var marker_8e0817c3784f37f1d8d0b4f2b165dd8c = L.marker(
                [10.861869812011719, 2.049091100692749],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_b32d505d27cc45cf89412759cdeb1028 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_8b550edeb60c6f32f182b4b71ec0a048 = $(`&lt;div id=&quot;html_8b550edeb60c6f32f182b4b71ec0a048&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KEROU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 94.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.049091100692749&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.861869812011719&lt;br&gt;&lt;/div&gt;`)[0];
                popup_b32d505d27cc45cf89412759cdeb1028.setContent(html_8b550edeb60c6f32f182b4b71ec0a048);



        marker_8e0817c3784f37f1d8d0b4f2b165dd8c.bindPopup(popup_b32d505d27cc45cf89412759cdeb1028)
        ;




            var marker_fa2525ce297848d215f7c29f4053bdfc = L.marker(
                [11.399999618530273, 2.180000066757202],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_b443a456358c874387530dc2e6b826ad = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_fec055b632e2100096f28657e3be2f16 = $(`&lt;div id=&quot;html_fec055b632e2100096f28657e3be2f16&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOMPONGOU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 82.4&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.180000066757202&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.399999618530273&lt;br&gt;&lt;/div&gt;`)[0];
                popup_b443a456358c874387530dc2e6b826ad.setContent(html_fec055b632e2100096f28657e3be2f16);



        marker_fa2525ce297848d215f7c29f4053bdfc.bindPopup(popup_b443a456358c874387530dc2e6b826ad)
        ;




            var marker_eaf1b040a3c346e44a943fdee0238410 = L.marker(
                [11.229999542236328, 2.6500000953674316],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_1b7bce6c19575ff57ac235b4842896f7 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_aeea1103c77c707e4904ff714fb11a76 = $(`&lt;div id=&quot;html_aeea1103c77c707e4904ff714fb11a76&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: ROUTE KANDI-BANIKOARA AMONT&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.6500000953674316&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.229999542236328&lt;br&gt;&lt;/div&gt;`)[0];
                popup_1b7bce6c19575ff57ac235b4842896f7.setContent(html_aeea1103c77c707e4904ff714fb11a76);



        marker_eaf1b040a3c346e44a943fdee0238410.bindPopup(popup_1b7bce6c19575ff57ac235b4842896f7)
        ;




            var marker_78dd4ab9578349603b8a1dbac3f9fb2f = L.marker(
                [11.229999542236328, 2.6500000953674316],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_22a9541ede6dc097dfaee0e8515f2dbc = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_9f069848e5bc89f799b21d0495958635 = $(`&lt;div id=&quot;html_9f069848e5bc89f799b21d0495958635&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: ROUTE KANDI-BANIKOARA AVAL&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 83.1&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.6500000953674316&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.229999542236328&lt;br&gt;&lt;/div&gt;`)[0];
                popup_22a9541ede6dc097dfaee0e8515f2dbc.setContent(html_9f069848e5bc89f799b21d0495958635);



        marker_78dd4ab9578349603b8a1dbac3f9fb2f.bindPopup(popup_22a9541ede6dc097dfaee0e8515f2dbc)
        ;




            var marker_2dbf31f47374c1fef141dd742574f765 = L.marker(
                [12.350000381469727, 2.7300000190734863],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_9c6a223b8c2a0422c3c512ddf50c7ecc = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_e6b0f148aaa30e809d870c027b6698a2 = $(`&lt;div id=&quot;html_e6b0f148aaa30e809d870c027b6698a2&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: BAROU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 2.7300000190734863&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 12.350000381469727&lt;br&gt;&lt;/div&gt;`)[0];
                popup_9c6a223b8c2a0422c3c512ddf50c7ecc.setContent(html_e6b0f148aaa30e809d870c027b6698a2);



        marker_2dbf31f47374c1fef141dd742574f765.bindPopup(popup_9c6a223b8c2a0422c3c512ddf50c7ecc)
        ;




            var marker_1ebb2c9ea2511db487c54cc71e33a5a5 = L.marker(
                [11.050000190734863, 3.049999952316284],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_1036864e06ad804f284ffc0d0ac686eb = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_a6159f5c436edf1e448f0861f59163c3 = $(`&lt;div id=&quot;html_a6159f5c436edf1e448f0861f59163c3&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: KOUTAKOUKROU&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 86.9&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 3.049999952316284&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.050000190734863&lt;br&gt;&lt;/div&gt;`)[0];
                popup_1036864e06ad804f284ffc0d0ac686eb.setContent(html_a6159f5c436edf1e448f0861f59163c3);



        marker_1ebb2c9ea2511db487c54cc71e33a5a5.bindPopup(popup_1036864e06ad804f284ffc0d0ac686eb)
        ;




            var marker_367540abb8f0195e95d9bf059ad93674 = L.marker(
                [11.866700172424316, 3.3833000659942627],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_09c10cdbf1fd7364d838ba229e0b7613 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_f3ef224429fe9e5bb5e530e99f1295a5 = $(`&lt;div id=&quot;html_f3ef224429fe9e5bb5e530e99f1295a5&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: MALANVILLE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 66.8&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 3.3833000659942627&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.866700172424316&lt;br&gt;&lt;/div&gt;`)[0];
                popup_09c10cdbf1fd7364d838ba229e0b7613.setContent(html_f3ef224429fe9e5bb5e530e99f1295a5);



        marker_367540abb8f0195e95d9bf059ad93674.bindPopup(popup_09c10cdbf1fd7364d838ba229e0b7613)
        ;




            var marker_6d454ae2f29c0e7dd1505c2e87959a61 = L.marker(
                [10.979999542236328, 3.25],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_7455a6117b37982a63cd4ba3c1856dd6 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_cf527e85be6119df5063b1b0938bc9ac = $(`&lt;div id=&quot;html_cf527e85be6119df5063b1b0938bc9ac&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: RTE KANDI-SEGBANA AVAL&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 97.8&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 3.25&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.979999542236328&lt;br&gt;&lt;/div&gt;`)[0];
                popup_7455a6117b37982a63cd4ba3c1856dd6.setContent(html_cf527e85be6119df5063b1b0938bc9ac);



        marker_6d454ae2f29c0e7dd1505c2e87959a61.bindPopup(popup_7455a6117b37982a63cd4ba3c1856dd6)
        ;




            var marker_01fddcd980f89105ab84cddef8c4a962 = L.marker(
                [10.979999542236328, 3.25],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_7cb77f5718d5783b5f4be36a40493056 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_5f24aea9d5aba2e589b047431aad13d8 = $(`&lt;div id=&quot;html_5f24aea9d5aba2e589b047431aad13d8&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: RTE KANDI-SEGBANA AMONT&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 100.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 3.25&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 10.979999542236328&lt;br&gt;&lt;/div&gt;`)[0];
                popup_7cb77f5718d5783b5f4be36a40493056.setContent(html_5f24aea9d5aba2e589b047431aad13d8);



        marker_01fddcd980f89105ab84cddef8c4a962.bindPopup(popup_7cb77f5718d5783b5f4be36a40493056)
        ;




            var marker_8f604f783974640ff26918955b4c69f9 = L.marker(
                [11.75, 3.3299999237060547],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_f1a01d005ab832925fc5cc9036ea8393 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_6cc45a6dfd501ff3d54c2ee94fff6d86 = $(`&lt;div id=&quot;html_6cc45a6dfd501ff3d54c2ee94fff6d86&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: COUBERI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 81.8&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 3.3299999237060547&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.75&lt;br&gt;&lt;/div&gt;`)[0];
                popup_f1a01d005ab832925fc5cc9036ea8393.setContent(html_6cc45a6dfd501ff3d54c2ee94fff6d86);



        marker_8f604f783974640ff26918955b4c69f9.bindPopup(popup_f1a01d005ab832925fc5cc9036ea8393)
        ;




            var marker_4d49048977d0a170b4f2998c1fb02b3b = L.marker(
                [7.800000190734863, 6.76669979095459],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_b4bab8b46f5677f1ceb05d048c8403a3 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_d8d2228ab2d78facb751cd10a15737ae = $(`&lt;div id=&quot;html_d8d2228ab2d78facb751cd10a15737ae&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: LOKOJA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 42.2&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 6.76669979095459&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 7.800000190734863&lt;br&gt;&lt;/div&gt;`)[0];
                popup_b4bab8b46f5677f1ceb05d048c8403a3.setContent(html_d8d2228ab2d78facb751cd10a15737ae);



        marker_4d49048977d0a170b4f2998c1fb02b3b.bindPopup(popup_b4bab8b46f5677f1ceb05d048c8403a3)
        ;




            var marker_8f6aa9ab75073a722ef134b57a35e70f = L.marker(
                [11.383299827575684, 4.133299827575684],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_4512b0c33a35a8eb2542c22891804c04 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_ade8e3254fbcb5f15dca946c4368fec4 = $(`&lt;div id=&quot;html_ade8e3254fbcb5f15dca946c4368fec4&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: YIDERE BODE&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 52.3&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 4.133299827575684&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 11.383299827575684&lt;br&gt;&lt;/div&gt;`)[0];
                popup_4512b0c33a35a8eb2542c22891804c04.setContent(html_ade8e3254fbcb5f15dca946c4368fec4);



        marker_8f6aa9ab75073a722ef134b57a35e70f.bindPopup(popup_4512b0c33a35a8eb2542c22891804c04)
        ;




            var marker_1e46db0d7aada7376f5fa0409473af16 = L.marker(
                [8.199999809265137, 9.73330020904541],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_0afb9a8ce48f81aa3e33c2bfde70f5ae = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_f7f0211d4e10db3e6f73575ca7785b77 = $(`&lt;div id=&quot;html_f7f0211d4e10db3e6f73575ca7785b77&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: IBI&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 75.0&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: 9.73330020904541&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 8.199999809265137&lt;br&gt;&lt;/div&gt;`)[0];
                popup_0afb9a8ce48f81aa3e33c2bfde70f5ae.setContent(html_f7f0211d4e10db3e6f73575ca7785b77);



        marker_1e46db0d7aada7376f5fa0409473af16.bindPopup(popup_0afb9a8ce48f81aa3e33c2bfde70f5ae)
        ;




            var marker_abd30d5ffcf1228e42d0d61ca8eb69f7 = L.marker(
                [14.079999923706055, -0.12999999523162842],
                {
}
            ).addTo(marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae);


        var popup_b719f5f40d3469e88d402e70500612d0 = L.popup({
  &quot;maxWidth&quot;: 200,
  &quot;minWidth&quot;: 100,
});



                var html_46ec9ee9fa78d349894c243235d6144c = $(`&lt;div id=&quot;html_46ec9ee9fa78d349894c243235d6144c&quot; style=&quot;width: 100.0%; height: 100.0%;&quot;&gt;&lt;b&gt;Station_name&lt;/b&gt;: YAKOUTA&lt;br&gt;&lt;b&gt;Percent_missing&lt;/b&gt;: 92.6&lt;br&gt;&lt;b&gt;Longitude&lt;/b&gt;: -0.12999999523162842&lt;br&gt;&lt;b&gt;Latitude&lt;/b&gt;: 14.079999923706055&lt;br&gt;&lt;/div&gt;`)[0];
                popup_b719f5f40d3469e88d402e70500612d0.setContent(html_46ec9ee9fa78d349894c243235d6144c);



        marker_abd30d5ffcf1228e42d0d61ca8eb69f7.bindPopup(popup_b719f5f40d3469e88d402e70500612d0)
        ;




            marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae.addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


            var layer_control_4ead7149ccb7792c758fbe5efc898691_layers = {
                base_layers : {
                    &quot;openstreetmap&quot; : tile_layer_a6c1f35601ac2f9e3811d50c058f037f,
                },
                overlays :  {
                    &quot;DEM&quot; : tile_layer_0401fe79e3b17c0ea9fb52edd4b15068,
                    &quot;Available water content&quot; : tile_layer_9c50beee1fd49223f644ea8a918cbf30,
                    &quot;Tree cover&quot; : tile_layer_4b7566c2434891991626134f7c96d735,
                    &quot;Slope&quot; : tile_layer_f2cc6261c50d7f64826a60bd81da39c0,
                    &quot;River network&quot; : tile_layer_4c542af7e89a3ccc700d66b07892850f,
                    &quot;Stations&quot; : marker_cluster_342e567b7ef6a85ed8b8c75869dd88ae,
                },
            };
            let layer_control_4ead7149ccb7792c758fbe5efc898691 = L.control.layers(
                layer_control_4ead7149ccb7792c758fbe5efc898691_layers.base_layers,
                layer_control_4ead7149ccb7792c758fbe5efc898691_layers.overlays,
                {
  &quot;position&quot;: &quot;topright&quot;,
  &quot;collapsed&quot;: true,
  &quot;autoZIndex&quot;: true,
}
            ).addTo(map_b85d2c3c819d33f59d25ee97b1f49321);


&lt;/script&gt;
&lt;/html&gt;" width="100%" height="600"style="border:none !important;" "allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen"></iframe>



#####   4. Training, Evaluating and Applying Bakaano-Hydro model 


```python
# INITIALIZE INSTANCE OF BAKAANO-HYDRO MODEL

from bakaano.runner import BakaanoHydro
bk = BakaanoHydro(  
    working_dir=working_dir, 
    study_area=study_area,
    climate_data_source='ERA5'
)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>




```python
# TRAINING BAKAANO-HYDRO MODEL

# The model is trained using the GRDC streamflow data.
# Note: The training process is computationally expensive and may take a long time to complete.
# trained model is always in the models folder in the working_dir and with a .keras extension
# the model names is always in the format: bakaano_model_<loss_fn>_<num_input_branch>_branches.keras

bk.train_streamflow_model(
    train_start='1991-01-01', 
    train_end='2020-12-31', 
    grdc_netcdf='/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-africa-south-america.nc', 
    lookback=365, 
    batch_size=1024, 
    num_epochs=100
)
```


```python
# EVALUATING THE TRAINED MODEL INTERACTIVELY

# The model is evaluated using the GRDC streamflow data.


# trained model is always in the models folder in the working_dir and with a .keras extension
# the model names is always in the format: bakaano_model_<loss_fn>_<num_input_branch>_branches.keras
model_path = f'{working_dir}/models/bakaano_model.keras' 

bk.evaluate_streamflow_model_interactively(
    model_path=model_path, 
    val_start='1981-01-01', 
    val_end='1988-12-31', 
    grdc_netcdf='/lustre/backup/WUR/ESG/duku002/NBAT/hydro/input_data/GRDC-Daily-africa-south-america.nc', 
    lookback=365
)
```



<style>
    .geemap-dark {
        --jp-widgets-color: white;
        --jp-widgets-label-color: white;
        --jp-ui-font-color1: white;
        --jp-layout-color2: #454545;
        background-color: #383838;
    }

    .geemap-dark .jupyter-button {
        --jp-layout-color3: #383838;
    }

    .geemap-colab {
        background-color: var(--colab-primary-surface-color, white);
    }

    .geemap-colab .jupyter-button {
        --jp-layout-color3: var(--colab-primary-surface-color, white);
    }
</style>



    Available station names:
    ['AKKA' 'ALCONGUI' 'ANSONGO' 'BADEGUICHERI' 'BANANKORO' 'BARO' 'BAROU'
     'BENENY-KEGNY' 'BOUGOUNI' 'CAMPEMENT DU DOUBLE VE' 'COUBERI' 'DABOLA'
     'DEBETE' 'DIALAKORO' 'DIOILA' 'DIONGORE AMONT' 'DIRE' 'DJIRILA' 'DOLBEL'
     'DOUNA' 'FARANAH' 'GARBE-KOUROU' 'GOUALA' 'GOUNDAM' 'GOUNOU-GAYA'
     'GUELELINKORO' 'GUINDAM ROUMDJI' 'GUINGUERINI' 'IBI' 'IRADOUGOU'
     'KAKASSI' 'KANDADJI' 'KANKAN' 'KARA' 'KE-MACINA' 'KEROU' 'KEROUANE'
     'KIRANGO AVAL' 'KISSIDOUGOU' 'KLELA' 'KOLONDIEBA' 'KOMPONGOU'
     'KONSANKORO' 'KORYOUME' 'KOULIKORO' 'KOUORO 1' 'KOUORO 2' 'KOUROUSSA'
     'KOUTAKOUKROU' 'KOUTO AMONT' 'KOUTO AVAL' 'LOKOJA' 'MADAROUNFA'
     'MADINA DIASSA' 'MALANVILLE' 'MANANKORO' 'MANDIANA' 'MOLOKORO' 'MOPTI'
     'NANTAKA (MOPTI)' 'NIAMEY' 'NIELLOUA' 'OUARAN' 'PANKOUROU' 'PAPARA'
     'PONONDOUGOU' 'PONT CAROL' 'ROUTE KANDI-BANIKOARA AMONT'
     'ROUTE KANDI-BANIKOARA AVAL' 'RTE KANDI-SEGBANA AMONT'
     'RTE KANDI-SEGBANA AVAL' 'SAMATIGUILA' 'SARAFERE' 'SELINGUE' 'SOFARA'
     'TAMOU' 'TERA' 'TIGUIBERY' 'TILEMBEYA' 'TINKISSO' 'TOMBOUGOU 1'
     'TOMBOUGOU 2' 'TONKA' 'TOSSAYE' 'TSERNAOUA' 'W' 'WAHIRE' 'YAKOUTA'
     'YANFOLILA' 'YIDERE BODE' 'ZIEMOUGOULA']
    INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:CPU:0',)


    2025-12-18 13:29:04.706041: E external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:152] failed call to cuInit: INTERNAL: CUDA error: Failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected


    [1m80/80[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m6s[0m 65ms/step
    Nash-Sutcliffe Efficiency (NSE): 0.7122612622507017
    Kling-Gupta Efficiency (KGE): 0.7443427177596089



    
![png](quick_start_files/quick_start_17_4.png)
    



```python
# PREDICTING STREAMFLOW USING THE TRAINED MODEL AND STORING AS CSV FILES 
# The model is used to predict streamflow in any location in the study area. 

model_path = f'{working_dir}/models/bakaano_model.keras'

bk.simulate_streamflow(
    model_path=model_path, 
    sim_start='1981-01-01', 
    sim_end='1988-12-31', 
    latlist=[13.8, 13.9, 9.15, 8.75, 10.66, 9.32, 7.8, 8.76, 6.17],
    lonlist=[3.0, 4.0, 4.77, 5.91, 4.69, 4.63, 8.91, 10.82, 6.77],
    lookback=365
)

```


