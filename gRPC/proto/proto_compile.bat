:: Directory of protobuf, should be installed alongside grpc
:: VCPKG_ROOT is the same as the one you set earlier
:: Worst case you can copy the whole path here starting from the vcpkg folder

:: set protobuf_cmd="%VCPKG_ROOT%/packages/protobuf_x64-windows/tools/protobuf/protoc"
set protobuf_cmd="C:\Users\<user>\Documents\VCPKG\vcpkg\packages\protobuf_x64-windows\tools\protobuf\protoc.exe"

::Directory where the grpc_cpp_plugin.exe can be found
::Other things are the same as above

::set grpc_exe_dir="%VCPKG_ROOT%/installed/x64-windows/tools/grpc/grpc_cpp_plugin.exe"
set grpc_exe_dir="C:\Users\<user>\Documents\VCPKG\vcpkg\installed\x64-windows\tools\grpc\grpc_cpp_plugin.exe"

::Folder where the source of the proto file
set src="%cd%"
::File to compile
::set proto_file="%cd%/hello.proto"
set proto_file="%cd%/training.proto"

::Folder where we dump the generated files
set dest="%cd%"

::The compile command itself
::%protobuf_cmd% --proto_path=%src% --cpp_out=%dest% --grpc_out=%dest% --plugin=protoc-gen-grpc=%grpc_exe_dir% %proto_file%

python -m grpc_tools.protoc --proto_path=%src% --python_out=%dest% --grpc_python_out=%dest% %proto_file%