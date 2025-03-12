import os

from .utils.utils_functions import tomcat_pipeline_cli
from typer import Typer, Context, Argument, Exit, Option
from typing_extensions import Annotated
from typing import List, Optional, Tuple
from rich import print
import numpy
from .. import __version__

'''----------------------------------------------'''
import logging
import sys

console_log_level = 10

logger = logging.getLogger(__name__)

console_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)
console_handler.setLevel(console_log_level)

DEBUG = 10
'''----------------------------------------------'''

# TODO : Need to change the examples CLI callouts that uses the dataset

app = Typer()


def __print_version(
        use_print: bool
) -> None:
    """Build-in function that prints the CLI version

    Args:
        use_print (Optional[bool]): flag to print the CLI version

    Returns:
        None

    """
    if use_print:
        print("CLI version : {}".format(__version__))
        raise Exit(code=0)


@app.callback(invoke_without_command=True)
def __cli_commands(
        cxt: Context,
        version: Annotated[Optional[bool], Option(
            "--version", "-v",
            help="application version",
            callback=__print_version,
            is_eager=True, is_flag=True)] = False
) -> None:
    """Function that prints all the possible functions to call using the deep

    Args:
        cxt (Context): context object
        version (Annotated[Optional[bool]): flag to print the CLI version

    Returns:
        None

    """
    # this part is to ensure to only print when the function is called without any command
    if cxt.invoked_subcommand:
        return

    sscraft_cli_mgs = """
                     _____          __  _   
                    |  __ \        / _|| |  
     ___  ___   ___ | |__) | __ _ | |_ | |_ 
    / __|/ __| / __||  _  / / _` ||  _|| __|
    \__ \\__ \| (__ | | \ \| (_| || |  | |_ 
    |___/|___/ \___||_|  \_\\__,_||_|   \__|
                                            
    """

    print("Welcome to the ssc-raft-CLI !!!\n")
    print(sscraft_cli_mgs)
    print("\nYou can use the followings commands to run the CLI : ")
    print(100 * "=")
    print("[blue][b]tomcat_recon[/]")
    print("\nwhere:")
    print("[blue]blue[/blue] pipeline for TOMCAT beamline reconstruction")

    print(100 * "=" + "\n")

    print("for more info about the functions, use the command ssc-raft function_name --help")

    
@app.command(name="tomcat_recon", help="TOMCAT beamline reconstruction pipeline function")
def tomcat_recon(
    data_name : Annotated[str, Argument(..., metavar="data_name", help="Name of the raw data hdf5 file")],
    dataPath : Annotated[str, Option(..., "--dataPath", "-dp", metavar="dataPath", help="Absolute data path")] = os.getcwd(),
    data_dataset : Annotated[str, Option(..., "--dataDataset", "-dd", metavar="dataDataset", help="Data hdf5 dataset")] = 'exchange/data',
    flat_path : Annotated[str, Option(..., "--flatPath", "-fp", metavar="flatPath", help="Absolute flat path")] = os.getcwd(),
    flat_dataset : Annotated[str, Option(..., "--flatDataset", "-fd", metavar="flatDataset", help="Flat hdf5 dataset")] = 'exchange/data_white_pre',
    dark_path : Annotated[str, Option(..., "--darkPath", "-da", metavar="darkPath", help="Absolute dark path")] = os.getcwd(),
    dark_dataset : Annotated[str, Option(..., "--darkDataset", "-dad", metavar="darkDataset", help="Dark hdf5 dataset")] = 'exchange/data_dark',
    recOutputDir : Annotated[str, Option(..., "--recOutputDir", "-O", metavar="recOutputDir", help="Absolute path of the reconstruction to be saved")] = os.getcwd(),
    OutID : Annotated[str, Option(..., "--OutID", "-Oid", metavar="OutID", help="Reconstruction ID name (user input)")] = 'default',
    numberOfGPUs : Annotated[int, Option(..., "--numberOfGPUs", "-ngpu", metavar="numberOfGPUs", help="Total number of gpus for reconstruction. Example for 2 GPUs: 2")] = 1,
    padding : Annotated[int, Option(..., "--padding", "-Z", metavar="padding", help="Data padding - Integer multiple of the data size (0,1,2, etc...)")] = 2,
    correctionType : Annotated[int, Option(..., "--correctionType", "-g", metavar="correctionType", help="Flat-dark normalization + log: option 7 available")] = 7,
    ringRemoval : Annotated[int, Option(..., "--ringRemoval", "-L", metavar="ringRemoval", help="Apply rings: 0=unused, 1=used")] = 1,
    ringMethod : Annotated[str, Option(..., "--ringMethod", "-Lm", metavar="ringMethod", help="Choose rings method. Options:  \'titarenko\'. Not implemented yet: \'all_stripes\', \'all_stripes_multiaxis\', \'none\'")] = 'titarenko',
    ringsLambda : Annotated[float, Option(..., "--ringLambda", "-Ll", metavar="ringsLambda", help="Rings: Titarenko regularization value")] = -1,
    ringsBlock : Annotated[int, Option(..., "--ringBlock", "-Lb", metavar="ringsBlock", help=" block value")] = 2,
    stripeRemoval : Annotated[Tuple[float,float,float], Option(..., "--stripeRemoval", "-sr", metavar="stripeRemoval", help="stripe removal in sinograms, takes 3 args: SNR, window size for filtering large stripes, window size for filtering small stripes (not used)")] = [10, 121, 21],
    centerOfRotation : Annotated[int, Option(..., "--centerOfRotation", "-c",  metavar="centerOfRotation", help="Center of rotation ")] = -1,
    stitching : Annotated[str, Option(..., "--stitching", "-S",  metavar="centestitchingOfRotation", help="Compute stitching. Option available: T = do stitching, F = no stitching")] = "F",
    stitchingOffset : Annotated[int, Option(..., "--stitchingOffset", "-So",  metavar="stitchingOffset", help="Offset for stitching")] = -1,
    paganinFilterParams : Annotated[Tuple[float,float,float,float,float], Option(..., "--paganinFilterParams", "-Y",  metavar="paganinFilterParams", help="paganin filter arguments, 5 entries: energy[keV],pixelSize[m],delta,beta,distance[m]")] = [1,1,1,0,1],
    paganinMethod : Annotated[str, Option(..., "--paganinMethod", "-Ym", metavar="paganinMethod", help="Choose Paganin method. Options: \'paganin_by_frames\', \'paganin_by_slices\', \'none\'")] = 'paganin_by_slices',
    reconstruct : Annotated[str, Option(..., "--reconstruct", "-R", metavar="reconstruct", help="do reconstruction of sinograms, argument is a string ID name of the output file in hdf5")] = 'Recon.h5',
    slices_recon : Annotated[Tuple[int,int], Option(..., "--slices", "-Rs", metavar="slices", help="Select slices to reconstruct, takes 2 args: start slice, end slice")] = [0, None],
    reconstructMethod : Annotated[str, Option(..., "--reconstructMethod", "-Rm", metavar="reconstructMethod", help="Choose the reconstruction method: \'fbp_RT\' and \'fbp_BST\'")] = "fbp_BST",
    filter : Annotated[str, Option(..., "--filter", "-F", metavar="filter", help="Choose reconstruction filter. Options: \'ramp\', \'gaussian\', \'hamming\', \'hann\', \'cosine\', \'lorentz\', \'rectangle\', \'none\'")] = "hamming",
    pinnedMem : Annotated[int, Option(..., "--pinnedMem", "-pm", metavar="pinnedMem", help="Use pinned memory on GPUs: 0=False, 1=True")] = 0

) -> None:

    """CLI function that apply TOMCAT tomography reconstruction pipeline
    
    To use the CLI function, use the following command
    ```{.sh title=help command}
    ssc-raft tomcat_recon --help
    ```

    ```{.sh title=output command}    

    Usage: ssc-raft tomcat_recon [OPTIONS] data_name                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                          
    TOMCAT beamline reconstruction pipeline function                                                                                                                                                                                                                                         

    ```                                                                                                                                                                                                                                                                                        
    ╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ *    data_name      data_name  Name of the raw data hdf5 file [default: None] [required]                                                                                                                                                                                               │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
    │ --dataPath             -dp        dataPath                  Absolute path of the data [default: /ibira/lnls/labs/tepui/home/paola.ferraz/projects/tomcat/CLI]                                                                                                                          │
    │ --recOutputDir         -O         recOutputDir              Absolute path of the reconstruction to be saved [default: /ibira/lnls/labs/tepui/home/paola.ferraz/projects/tomcat/CLI]                                                                                                    │
    │ --numberOfGPUs         -ngpu      numberOfGPUs              Total number of gpus for reconstruction. Example for 2 GPUs: 2 [default: 1]                                                                                                                                                │
    │ --zeroPadding          -Z         zeroPadding               Data padding - Integer multiple of the data size (0,1,2, etc...) [default: 0]                                                                                                                                              │
    │ --correctionType       -g         correctionType            Flat-dark normalization + log: option 7 available [default: 7]                                                                                                                                                             │
    │ --ringRemoval          -L         ringRemoval               Apply rings: 0=unused, 1=used [default: 1]                                                                                                                                                                                 │
    │ --ringMethod           -Lm        ringMethod                Choose rings method. Options:  'titarenko'. Not implemented yet: 'all_stripes', 'all_stripes_multiaxis', 'none' [default: titarenko]                                                                                       │
    │ --ringLambda           -Ll        ringsLambda               Rings: Titarenko regularization value [default: -1]                                                                                                                                                                        │
    │ --ringBlock            -Lb        ringsBlock                block value [default: 2]                                                                                                                                                                                                   │
    │ --stripeRemoval        -sr        stripeRemoval             stripe removal in sinograms, takes either 1 or 3 args. 1 arg: window size of the median filter. 3 args: SNR, window size for filtering large stripes, window size for filtering small stripes (not used)                   │
    │                                                             [default: 10, 121, 21]                                                                                                                                                                                                     │
    │ --centerOfRotation     -c         centerOfRotation          Center of rotation or overlap for stitching [default: -1]                                                                                                                                                                  │
    │ --stitching            -S         centestitchingOfRotation  Compute stitching. Option available: T = do stitching, F = no stitching [default: F]                                                                                                                                       │
    │ --paganinFilterParams  -Y         paganinFilterParams       paganin filter argument string, needs: energy[keV],pixelSize[m],delta,beta,distance[m] [default: 1.0, 1.0, 1.0, 0.0, 1.0]                                                                                                            │
    │ --paganinMethod        -Ym        paganinMethod             Choose Paganin method. Options: 'paganin_by_frames', 'paganin_by_slices', 'none' [default: paganin_by_slices]                                                                                                              │
    │ --reconstruct          -R         reconstruct               do reconstruction of sinograms, argument is a string ID name of the output file in hdf5 [default: Recon_]                                                                                                                  │
    │ --reconstructMethod    -Rm        reconstructMethod         Choose the reconstruction method: 'fbp_RT' and 'fbp_BST' [default: fbp_BST]                                                                                                                                                │
    │ --filter               -F         filter                    Choose reconstruction filter. Options: 'ramp', 'gaussian', 'hamming', 'hann', 'cosine', 'lorentz', 'rectangle', 'none' [default: hamming]                                                                                  │
    │ --pinnedMem            -pm        pinnedMem                 Use pinned memory on GPUs: 0=False, 1=True [default: 0]                                                                                                                                                                    │
    │ --help                                                      Show this message and exit.                                                                                                                                                                                                │
    ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
    ```
                                                                                                                                                                                                                                                                                      
    """

    gpus_list = [i for i in range(numberOfGPUs)]

    if slices_recon[1] is None:
        slices_recon = None

    dic = {
        "pin_memory":pinnedMem,
        "input_path": dataPath, 
        "input_name": data_name, 
        "input_data_hdf5_path": data_dataset,
        "flat_path": os.path.join(dataPath,data_name),
        "flat_pre_dataset": flat_dataset,
        "flat_pos_dataset": 'exchange/data_white_post',
        "dark_path": os.path.join(dataPath,data_name),
        "dark_dataset":dark_dataset,
        "output_path": recOutputDir,
        "id": OutID,
        "gpu": gpus_list,  
        "paganin_method": paganinMethod,
        "z2[m]": paganinFilterParams[4],
        "detectorPixel[m]": paganinFilterParams[1],
        "energy[eV]": paganinFilterParams[0]*1e3,
        "beta/delta": paganinFilterParams[3]/paganinFilterParams[2],
        "padding": padding,
        "correction": correctionType, 
        "rings": ringRemoval,
        "rings_method": ringMethod, 
        "rings_regularization": ringsLambda, 
        "rings_block": ringsBlock, 
        "rings_parameters": stripeRemoval, 
        "stitching":stitching,
        "stitching overlap": stitchingOffset,
        "axis offset": centerOfRotation, 
        "reconstruct": reconstruct, 
        "slices": slices_recon,
        "method": reconstructMethod, 
        "filter": filter,
        "save_recon": True,
        "crop_circle_recon":False,
        "blocksize": 0
    }

    dic['reconstruct'] = 'Recon_' + OutID + '_' + data_name

    tomcat_pipeline_cli(dic)

if __name__ == "__main__":
    app()
