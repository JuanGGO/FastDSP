FILE(GLOB NPPI_LIBS /usr/local/cuda-10.2/lib64/*npp*)
add_library(fdspalgorithms fft.cu image_processing/morphology.cu reductions.cu)
target_link_libraries(fdspalgorithms ${NPPI_LIBS})