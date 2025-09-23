>> create a TODO list when working on complex tasks to track progress and remain on track
>> note: for every modified code, run [py_compile, function calling, IMPORT calling, Redundancy, etc] tests >> View the results >> before making any claim about improvements
>> in virtual environment {venv_<project_name>}, dont install packages individually at any cost
    >> note: example modify [ requirements.txt ] >> numpy==1.26.
    >> then [source venv_<project_name>/bin/activate] && [ pip install -r requirements ]
    >> dont stop untill you resolve all errors
    >> but dont install packages individually at any cost , do you understand? 
>> if image or its directory is shared, view each of the following image with FRESH eyes, before making any conclusion
>> re-read ALL lines of current version of code, before making any modification 
>> do NOT HARDCODE or do NOT use FALLBACK [e.g - CPU, if OOM on GPU] mechanism at any cost
    >> [if its difficult to implement, user's requirement, explicitly say so]
>> create TEST file in [ unit_test/ ] ONLY existing directory  so that i cant test it manually outside claude terminal, if execution_time > 2 mins
>> do not be YES MAN >>  give me PROS and CONS of atleast 3  approaches 
>> > make sure all outputs [images, ] from all scripts are stored inside one centralized folder
  >> say [ outputs/ ] directory >>
  >> you may create individual folder for each python file inside [ outputs/ ] directory

  >> also all log / data generated should be stored in [outputs/centralized.db] single datasource
  >> you may keep individual TABLE for each python file
  >> so that next python files can use the output of previous python file directly from [outputs/centralized.db] single datasource
  > avoid creating txt/json file for logs >> so we  wont have to do text matching  
>> keep all python files in [ src/ ] directory
>> keep SAME names for python files in [ src/ ] directory 
    and their respective sub folder in [ outputs/ ] directory
    and their respctive "table"s with same name in [ outputs/centralize.db ]
    >> Zero-padded numbering: "m01_<name>", "m02_<name>", "m03_<name>" (ensures proper sorting)
      for python files in [ src/ ] directory
      for respective  subfolder in [ outputs/ ] directory
      for respective tables in [ outputs/centralized.db ] database
> before building  next code modules >> READ, analyze and explain the output of previous code module