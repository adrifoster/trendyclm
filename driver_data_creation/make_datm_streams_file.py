import os
import glob

def write_to_file(text, file_out):
    """
    Writes text to a file, surrounding text with \n characters
    """
    file_out.write("\n{}\n".format(text))

def write_datm_streams_lines(streamname, datmfiles, mesh_file, file, varlist=None):
    """
    writes out lines for the user_nl_datm_streams file for a specific DATM stream

    streamname - stream name (e.g. TPQW)
    datmfiles - comma-separated list (str) of DATM file names
    file - file connection to user_nl_datm_streams file
    """
    write_to_file("{}:datafiles={}".format(streamname, ",".join(datmfiles)), file)
    write_to_file("{}:meshfile={}".format(streamname, mesh_file), file)
    
    if varlist is not None:
        write_to_file(f"{streamname}:datavars= {', '.join([' '.join(var) for var in varlist])}", file)

def get_file_names(dir, file_prefix, stream_name):
    return sorted(glob.glob(f"{dir}/{file_prefix}.{stream_name}.*.nc"))

def main(dir, file_prefix, out_dir, mesh_file, use_diffuse, solr_vars, tpqwl_vars):
    
    prec_prefix = "Prec"
    tpqwl_prefix = "TPQWL"
    if use_diffuse:
        solr_prefix='Solr_Diffuse'
    else:
        solr_prefix='Solr'
    
    prec_files = get_file_names(dir, file_prefix, prec_prefix)
    tpqwl_files = get_file_names(dir, file_prefix, tpqwl_prefix)
    solr_files = get_file_names(dir, file_prefix, solr_prefix)
    
    nl_file = 'user_nl_datm_streams'
    
    with open(os.path.join(out_dir, nl_file), "w") as file:
        write_datm_streams_lines("CLMCRUNCEPv7.Solar", solr_files, mesh_file, file, solr_vars)
        write_datm_streams_lines("CLMCRUNCEPv7.Precip", prec_files, mesh_file, file)
        write_datm_streams_lines("CLMCRUNCEPv7.TPQW", tpqwl_files, mesh_file, file, tpqwl_vars)


if __name__ == '__main__':
    dir = '/glade/campaign/cgd/tss/projects/TRENDY2025/inputs/three_stream'
    file_prefix = 'clmforc.CRUJRAv3_0.5x0.5'
    out_dir = '/glade/work/afoster/TRENDY_2025/user_mods'
    mesh_file = '/glade/campaign/cgd/tss/projects/TRENDY2025/inputs/three_stream/ESMFmesh_CRUJRAv2.5_c2024.5d_mask_240722.nc'
    solr_vars = [['FSDS_DIRECT', 'Faxa_swdndr'], ['FSDS_DIFFUSE', 'Faxa_swdndf']]
    tpqwl_vars = [['QBOT', 'Sa_shum'], ['PSRF', 'Sa_pbot'], ['TBOT', 'Sa_tbot'], ['WIND', 'Sa_wind'], ['FLDS', 'Faxa_lwdn']]
    main(dir, file_prefix, out_dir, mesh_file, True, solr_vars, tpqwl_vars)