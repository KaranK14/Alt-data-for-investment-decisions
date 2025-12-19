"""
Step 1: Parse raw ClinicalTrials.gov JSON data from v2 API download.

Reads from JSONL files created by download_oncology_trials_v2.py:
- oncology_phase234_interventional_industry_full_raw.jsonl (preferred - full records)
- Falls back to oncology_phase234_interventional_industry_search_raw.jsonl

Filters for:
- Phase 2, 3, or 4 (including 2/3 combo)
- Interventional
- Industry-sponsored
- All oncology indications

Extracts comprehensive variables from all available JSON modules.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add Project 2 directory to path so we can import scripts
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config import (
    DATA_DIR,
)


def safe_get(data: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dictionary."""
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def parse_date(date_struct: Optional[Dict]) -> Optional[str]:
    """Parse date structure from API response."""
    if isinstance(date_struct, dict):
        return safe_get(date_struct, 'date')
    return None


def parse_trial_record(record: Dict, nct_id: str = None) -> Optional[Dict[str, Any]]:
    """
    Parse a single ClinicalTrials.gov study record from JSONL.
    
    Args:
        record: Full study record dictionary from JSONL
        nct_id: NCT ID (may be in record or passed separately)
    
    Returns:
        Dictionary with extracted variables, or None if trial doesn't match filters
    """
    # Get nct_id if not provided
    if not nct_id:
        protocol = safe_get(record, 'protocolSection', {})
        id_module = safe_get(protocol, 'identificationModule', {})
        nct_id = safe_get(id_module, 'nctId')
    
    if not nct_id:
        return None
    
    protocol = safe_get(record, 'protocolSection', {})
    if not protocol:
        return None
    
    # ========================================================================
    # FILTERING
    # ========================================================================
    
    # Check phase (Phase 2, 3, or 4)
    design_module = safe_get(protocol, 'designModule', {})
    phases = safe_get(design_module, 'phases', default=[])
    if isinstance(phases, list):
        valid_phases = {"PHASE2", "PHASE3", "PHASE4", "PHASE2|PHASE3"}
        if not any(phase in valid_phases for phase in phases):
            return None
    else:
        if phases not in {"PHASE2", "PHASE3", "PHASE4", "PHASE2|PHASE3"}:
            return None
    
    # Check study type (Interventional)
    study_type = safe_get(design_module, 'studyType', '').upper()
    if study_type != 'INTERVENTIONAL':
        return None
    
    # Check sponsor type (Industry)
    sponsor_module = safe_get(protocol, 'sponsorCollaboratorsModule', {})
    lead_sponsor = safe_get(sponsor_module, 'leadSponsor', default={})
    sponsor_type = safe_get(lead_sponsor, 'class', '').upper()
    if sponsor_type != 'INDUSTRY':
        return None
    
    # ========================================================================
    # EXTRACT VARIABLES - COMPREHENSIVE EXTRACTION
    # ========================================================================
    
    result = {}
    result['nct_id'] = nct_id
    
    # ========================================================================
    # IDENTIFICATION MODULE
    # ========================================================================
    identification_module = safe_get(protocol, 'identificationModule', {})
    result['brief_title'] = safe_get(identification_module, 'briefTitle')
    result['official_title'] = safe_get(identification_module, 'officialTitle')
    result['acronym'] = safe_get(identification_module, 'acronym')
    
    # Secondary IDs
    secondary_ids = safe_get(identification_module, 'secondaryIdInfos', default=[])
    if isinstance(secondary_ids, list):
        secondary_id_list = []
        for sid in secondary_ids:
            if isinstance(sid, dict):
                sid_value = safe_get(sid, 'id') or safe_get(sid, 'value')
                if sid_value:
                    secondary_id_list.append(sid_value)
        result['secondary_ids'] = json.dumps(secondary_id_list) if secondary_id_list else None
    else:
        result['secondary_ids'] = None
    
    # ========================================================================
    # STATUS MODULE
    # ========================================================================
    status_module = safe_get(protocol, 'statusModule', {})
    result['overall_status'] = safe_get(status_module, 'overallStatus')
    result['last_known_status'] = safe_get(status_module, 'lastKnownStatus')
    result['why_stopped'] = safe_get(status_module, 'whyStopped')
    result['status_verified_date'] = safe_get(status_module, 'statusVerifiedDate')
    
    # Dates
    result['start_date'] = parse_date(safe_get(status_module, 'startDateStruct'))
    result['primary_completion_date'] = parse_date(safe_get(status_module, 'primaryCompletionDateStruct'))
    result['completion_date'] = parse_date(safe_get(status_module, 'completionDateStruct'))
    result['study_first_submit_date'] = parse_date(safe_get(status_module, 'studyFirstSubmitDate'))
    result['study_first_submit_qc_date'] = parse_date(safe_get(status_module, 'studyFirstSubmitQcDate'))
    result['first_posted_date'] = parse_date(safe_get(status_module, 'studyFirstPostDateStruct'))
    result['results_first_submit_date'] = parse_date(safe_get(status_module, 'resultsFirstSubmitDate'))
    result['results_first_submit_qc_date'] = parse_date(safe_get(status_module, 'resultsFirstSubmitQcDate'))
    result['results_first_posted_date'] = parse_date(safe_get(status_module, 'resultsFirstPostDateStruct'))
    result['last_update_submit_date'] = parse_date(safe_get(status_module, 'lastUpdateSubmitDate'))
    result['last_update_posted_date'] = parse_date(safe_get(status_module, 'lastUpdatePostDateStruct'))
    
    # Expanded access
    expanded_access = safe_get(status_module, 'expandedAccessInfo', {})
    result['has_expanded_access'] = safe_get(expanded_access, 'hasExpandedAccess')
    
    # Enrollment info from status
    enrollment_info_status = safe_get(status_module, 'enrollmentInfo', default={})
    if isinstance(enrollment_info_status, dict):
        result['enrollment_actual'] = safe_get(enrollment_info_status, 'actualCount')
        result['enrollment_actual_type'] = safe_get(enrollment_info_status, 'type')
    else:
        result['enrollment_actual'] = None
        result['enrollment_actual_type'] = None
    
    # ========================================================================
    # DESIGN MODULE
    # ========================================================================
    result['study_type'] = safe_get(design_module, 'studyType')
    
    # Phases
    if isinstance(phases, list):
        result['phase'] = phases[0] if phases else None
        result['phases_list'] = json.dumps(phases) if phases else None
    else:
        result['phase'] = phases
        result['phases_list'] = None
    
    design_info = safe_get(design_module, 'designInfo', default={})
    if isinstance(design_info, dict):
        result['allocation'] = safe_get(design_info, 'allocation')
        result['intervention_model'] = safe_get(design_info, 'interventionModel')
        result['intervention_model_description'] = safe_get(design_info, 'interventionModelDescription')
        result['primary_purpose'] = safe_get(design_info, 'primaryPurpose')
        result['observational_model'] = safe_get(design_info, 'observationalModel')
        result['time_perspective'] = safe_get(design_info, 'timePerspective')
        
        masking_info = safe_get(design_info, 'maskingInfo', default={})
        if isinstance(masking_info, dict):
            result['masking'] = safe_get(masking_info, 'masking')
            result['masking_description'] = safe_get(masking_info, 'maskingDescription')
            who_masked = safe_get(masking_info, 'whoMasked', default=[])
            if isinstance(who_masked, list):
                result['who_masked'] = json.dumps(who_masked) if who_masked else None
            else:
                result['who_masked'] = None
        else:
            result['masking'] = None
            result['masking_description'] = None
            result['who_masked'] = None
        
        result['patient_registry'] = safe_get(design_info, 'patientRegistry')
        result['target_duration'] = safe_get(design_module, 'targetDuration')
    else:
        result['allocation'] = None
        result['intervention_model'] = None
        result['intervention_model_description'] = None
        result['primary_purpose'] = None
        result['masking'] = None
        result['masking_description'] = None
        result['who_masked'] = None
        result['patient_registry'] = None
        result['target_duration'] = None
    
    # Enrollment info from design
    enrollment_info_design = safe_get(design_module, 'enrollmentInfo', default={})
    if isinstance(enrollment_info_design, dict):
        result['enrollment_planned'] = safe_get(enrollment_info_design, 'count')
        result['enrollment_planned_type'] = safe_get(enrollment_info_design, 'type')
    else:
        if result.get('enrollment_planned') is None:
            result['enrollment_planned'] = None
            result['enrollment_planned_type'] = None
    
    # BioSpec
    bio_spec = safe_get(design_module, 'bioSpec', {})
    result['biospec_retention'] = safe_get(bio_spec, 'retention')
    result['biospec_description'] = safe_get(bio_spec, 'description')
    
    # ========================================================================
    # ARMS & INTERVENTIONS MODULE
    # ========================================================================
    arms_interventions = safe_get(protocol, 'armsInterventionsModule', {})
    
    # Arms
    arms = safe_get(arms_interventions, 'armGroups', default=[])
    if isinstance(arms, list):
        result['number_of_arms'] = len(arms)
        arm_details = []
        for arm in arms:
            if isinstance(arm, dict):
                arm_details.append({
                    'label': safe_get(arm, 'label'),
                    'type': safe_get(arm, 'type'),
                    'description': safe_get(arm, 'description'),
                    'interventionNames': safe_get(arm, 'interventionNames', default=[])
                })
        result['arm_groups'] = json.dumps(arm_details) if arm_details else None
    else:
        result['number_of_arms'] = 0
        result['arm_groups'] = None
    
    # Interventions
    interventions = safe_get(arms_interventions, 'interventions', default=[])
    if isinstance(interventions, list):
        intervention_list = []
        intervention_names = []
        intervention_types = []
        for interv in interventions:
            if isinstance(interv, dict):
                interv_name = safe_get(interv, 'name')
                interv_type = safe_get(interv, 'type')
                interv_desc = safe_get(interv, 'description')
                
                if interv_name:
                    intervention_names.append(interv_name)
                if interv_type:
                    intervention_types.append(interv_type)
                
                intervention_list.append({
                    'name': interv_name,
                    'type': interv_type,
                    'description': interv_desc
                })
        
        result['interventions'] = json.dumps(intervention_list) if intervention_list else None
        result['intervention_names'] = json.dumps(intervention_names) if intervention_names else None
        result['intervention_types'] = json.dumps(intervention_types) if intervention_types else None
        result['intervention_text'] = ' '.join(intervention_names) if intervention_names else None
    else:
        result['interventions'] = None
        result['intervention_names'] = None
        result['intervention_types'] = None
        result['intervention_text'] = None
    
    # ========================================================================
    # CONDITIONS MODULE
    # ========================================================================
    conditions_module = safe_get(protocol, 'conditionsModule', {})
    conditions = safe_get(conditions_module, 'conditions', default=[])
    if isinstance(conditions, list):
        result['condition_list'] = json.dumps(conditions) if conditions else None
        result['condition_text'] = ' '.join(conditions) if conditions else None
    else:
        result['condition_list'] = None
        result['condition_text'] = None
    
    keywords = safe_get(conditions_module, 'keywords', default=[])
    if isinstance(keywords, list):
        result['keywords'] = json.dumps(keywords) if keywords else None
    else:
        result['keywords'] = None
    
    # ========================================================================
    # OUTCOMES MODULE
    # ========================================================================
    outcomes_module = safe_get(protocol, 'outcomesModule', {})
    
    # Primary outcomes
    primary_outcomes = safe_get(outcomes_module, 'primaryOutcomes', default=[])
    if isinstance(primary_outcomes, list):
        primary_measures = []
        primary_details = []
        for outcome in primary_outcomes:
            if isinstance(outcome, dict):
                measure = safe_get(outcome, 'measure')
                description = safe_get(outcome, 'description')
                time_frame = safe_get(outcome, 'timeFrame')
                
                if measure:
                    primary_measures.append(measure)
                
                primary_details.append({
                    'measure': measure,
                    'description': description,
                    'timeFrame': time_frame
                })
        result['primary_outcome_measures'] = json.dumps(primary_measures) if primary_measures else None
        result['primary_outcomes'] = json.dumps(primary_details) if primary_details else None
    else:
        result['primary_outcome_measures'] = None
        result['primary_outcomes'] = None
    
    # Secondary outcomes
    secondary_outcomes = safe_get(outcomes_module, 'secondaryOutcomes', default=[])
    if isinstance(secondary_outcomes, list):
        secondary_measures = []
        secondary_details = []
        for outcome in secondary_outcomes:
            if isinstance(outcome, dict):
                measure = safe_get(outcome, 'measure')
                description = safe_get(outcome, 'description')
                time_frame = safe_get(outcome, 'timeFrame')
                
                if measure:
                    secondary_measures.append(measure)
                
                secondary_details.append({
                    'measure': measure,
                    'description': description,
                    'timeFrame': time_frame
                })
        result['secondary_outcome_measures'] = json.dumps(secondary_measures) if secondary_measures else None
        result['secondary_outcomes'] = json.dumps(secondary_details) if secondary_details else None
    else:
        result['secondary_outcome_measures'] = None
        result['secondary_outcomes'] = None
    
    # Other outcomes
    other_outcomes = safe_get(outcomes_module, 'otherOutcomes', default=[])
    if isinstance(other_outcomes, list):
        other_measures = []
        for outcome in other_outcomes:
            if isinstance(outcome, dict):
                measure = safe_get(outcome, 'measure') or safe_get(outcome, 'description')
                if measure:
                    other_measures.append(measure)
        result['other_outcome_measures'] = json.dumps(other_measures) if other_measures else None
    else:
        result['other_outcome_measures'] = None
    
    # ========================================================================
    # ELIGIBILITY MODULE
    # ========================================================================
    eligibility_module = safe_get(protocol, 'eligibilityModule', {})
    result['eligibility_criteria_text'] = safe_get(eligibility_module, 'eligibilityCriteria')
    result['study_population'] = safe_get(eligibility_module, 'studyPopulation')
    result['sampling_method'] = safe_get(eligibility_module, 'samplingMethod')
    result['healthy_volunteers'] = safe_get(eligibility_module, 'healthyVolunteers')
    result['sex'] = safe_get(eligibility_module, 'sex')
    result['gender'] = safe_get(eligibility_module, 'sex')  # Alias
    result['gender_based'] = safe_get(eligibility_module, 'genderBased')
    result['gender_description'] = safe_get(eligibility_module, 'genderDescription')
    result['min_age'] = safe_get(eligibility_module, 'minimumAge')
    result['max_age'] = safe_get(eligibility_module, 'maximumAge')
    result['std_ages'] = safe_get(eligibility_module, 'stdAges', default=[])
    if isinstance(result['std_ages'], list):
        result['std_ages'] = json.dumps(result['std_ages']) if result['std_ages'] else None
    
    # Enrollment info from eligibility
    enrollment_info_eligibility = safe_get(eligibility_module, 'enrollmentInfo', default={})
    if isinstance(enrollment_info_eligibility, dict):
        if result.get('enrollment_planned') is None:
            result['enrollment_planned'] = safe_get(enrollment_info_eligibility, 'count')
            result['enrollment_planned_type'] = safe_get(enrollment_info_eligibility, 'type')
    
    # ========================================================================
    # SPONSOR & COLLABORATORS MODULE
    # ========================================================================
    lead_sponsor = safe_get(sponsor_module, 'leadSponsor', default={})
    if isinstance(lead_sponsor, dict):
        result['lead_sponsor_name'] = safe_get(lead_sponsor, 'name')
        sponsor_class = safe_get(lead_sponsor, 'class', '').upper()
        result['lead_sponsor_type'] = 'Industry' if sponsor_class == 'INDUSTRY' else 'Other'
    else:
        result['lead_sponsor_name'] = None
        result['lead_sponsor_type'] = None
    
    # Collaborators
    collaborators = safe_get(sponsor_module, 'collaborators', default=[])
    if isinstance(collaborators, list):
        collaborator_list = []
        for collab in collaborators:
            if isinstance(collab, dict):
                collaborator_list.append({
                    'name': safe_get(collab, 'name'),
                    'class': safe_get(collab, 'class')
                })
        result['collaborators'] = json.dumps(collaborator_list) if collaborator_list else None
    else:
        result['collaborators'] = None
    
    # Responsible party
    responsible_party = safe_get(sponsor_module, 'responsibleParty', default={})
    if isinstance(responsible_party, dict):
        result['responsible_party_type'] = safe_get(responsible_party, 'type')
        result['responsible_party_name'] = safe_get(responsible_party, 'name') or safe_get(responsible_party, 'oldNameTitle')
        result['responsible_party_title'] = safe_get(responsible_party, 'oldNameTitle')
        result['responsible_party_organization'] = safe_get(responsible_party, 'oldOrganization')
        result['responsible_party_investigator_full_name'] = safe_get(responsible_party, 'investigatorFullName')
        result['responsible_party_investigator_title'] = safe_get(responsible_party, 'investigatorTitle')
        result['responsible_party_investigator_affiliation'] = safe_get(responsible_party, 'investigatorAffiliation')
    else:
        result['responsible_party_type'] = None
        result['responsible_party_name'] = None
        result['responsible_party_title'] = None
        result['responsible_party_organization'] = None
        result['responsible_party_investigator_full_name'] = None
        result['responsible_party_investigator_title'] = None
        result['responsible_party_investigator_affiliation'] = None
    
    # ========================================================================
    # CONTACTS & LOCATIONS MODULE
    # ========================================================================
    contacts_module = safe_get(protocol, 'contactsLocationsModule', {})
    
    # Locations
    locations = safe_get(contacts_module, 'locations', default=[])
    if isinstance(locations, list):
        result['number_of_facilities'] = len(locations)
        facility_countries = []
        facility_cities = []
        facility_states = []
        for loc in locations:
            if isinstance(loc, dict):
                country = safe_get(loc, 'country')
                city = safe_get(loc, 'city')
                state = safe_get(loc, 'state')
                
                if country and country not in facility_countries:
                    facility_countries.append(str(country))
                if city and city not in facility_cities:
                    facility_cities.append(str(city))
                if state and state not in facility_states:
                    facility_states.append(str(state))
        
        result['facility_countries'] = json.dumps(facility_countries) if facility_countries else None
        result['facility_cities'] = json.dumps(facility_cities) if facility_cities else None
        result['facility_states'] = json.dumps(facility_states) if facility_states else None
        
        # Create flags
        result['has_us_sites'] = 1 if 'United States' in facility_countries else 0
        result['has_china_sites'] = 1 if 'China' in facility_countries else 0
        result['has_eu_sites'] = 1 if any(c in facility_countries for c in ['Germany', 'France', 'Italy', 'Spain', 'United Kingdom']) else 0
        result['is_multicountry'] = 1 if len(facility_countries) > 1 else 0
    else:
        result['number_of_facilities'] = 0
        result['facility_countries'] = None
        result['facility_cities'] = None
        result['facility_states'] = None
        result['has_us_sites'] = 0
        result['has_china_sites'] = 0
        result['has_eu_sites'] = 0
        result['is_multicountry'] = 0
    
    # Overall officials
    overall_officials = safe_get(contacts_module, 'overallOfficials', default=[])
    if isinstance(overall_officials, list):
        officials_list = []
        for official in overall_officials:
            if isinstance(official, dict):
                officials_list.append({
                    'name': safe_get(official, 'name'),
                    'role': safe_get(official, 'role'),
                    'affiliation': safe_get(official, 'affiliation')
                })
        result['overall_officials'] = json.dumps(officials_list) if officials_list else None
    else:
        result['overall_officials'] = None
    
    # Central contacts
    central_contacts = safe_get(contacts_module, 'centralContacts', default=[])
    if isinstance(central_contacts, list):
        contacts_list = []
        for contact in central_contacts:
            if isinstance(contact, dict):
                contacts_list.append({
                    'name': safe_get(contact, 'name'),
                    'role': safe_get(contact, 'role'),
                    'phone': safe_get(contact, 'phone'),
                    'email': safe_get(contact, 'email')
                })
        result['central_contacts'] = json.dumps(contacts_list) if contacts_list else None
    else:
        result['central_contacts'] = None
    
    # ========================================================================
    # DESCRIPTION MODULE
    # ========================================================================
    description_module = safe_get(protocol, 'descriptionModule', {})
    result['brief_summary'] = safe_get(description_module, 'briefSummary')
    result['detailed_description'] = safe_get(description_module, 'detailedDescription')
    
    # Combined description text for embeddings
    description_parts = [
        result.get('brief_title', ''),
        result.get('official_title', ''),
        result.get('brief_summary', ''),
        result.get('detailed_description', ''),
        result.get('eligibility_criteria_text', '')
    ]
    result['description_text'] = ' '.join(filter(None, description_parts))
    
    # ========================================================================
    # OVERSIGHT MODULE (if available)
    # ========================================================================
    oversight_module = safe_get(protocol, 'oversightModule', {}) or safe_get(protocol, 'studyOversightModule', {})
    if isinstance(oversight_module, dict):
        result['has_dmc'] = safe_get(oversight_module, 'dataMonitoringCommittee') or safe_get(oversight_module, 'oversightHasDmc')
        result['is_fda_regulated_drug'] = safe_get(oversight_module, 'isFdaRegulatedDrug')
        result['is_fda_regulated_device'] = safe_get(oversight_module, 'isFdaRegulatedDevice')
        result['is_unapproved_device'] = safe_get(oversight_module, 'isUnapprovedDevice')
        result['is_ppsd'] = safe_get(oversight_module, 'isPpsd')
        result['is_us_export'] = safe_get(oversight_module, 'isUsExport')
    else:
        result['has_dmc'] = None
        result['is_fda_regulated_drug'] = None
        result['is_fda_regulated_device'] = None
        result['is_unapproved_device'] = None
        result['is_ppsd'] = None
        result['is_us_export'] = None
    
    # ========================================================================
    # IPD SHARING STATEMENT MODULE
    # ========================================================================
    ipd_module = safe_get(protocol, 'ipdSharingStatementModule', {})
    if isinstance(ipd_module, dict):
        ipd_sharing = safe_get(ipd_module, 'ipdSharing')
        if ipd_sharing and isinstance(ipd_sharing, str):
            result['has_ipd_sharing_plan'] = 1 if ipd_sharing.upper() == 'YES' else 0
        else:
            result['has_ipd_sharing_plan'] = 0
        
        result['ipd_sharing_description'] = safe_get(ipd_module, 'description')
        result['ipd_sharing_info_types'] = json.dumps(safe_get(ipd_module, 'infoTypes', default=[]))
        result['ipd_sharing_time_frame'] = safe_get(ipd_module, 'timeFrame')
        result['ipd_sharing_access_criteria'] = safe_get(ipd_module, 'accessCriteria')
        result['ipd_sharing_url'] = safe_get(ipd_module, 'url')
    else:
        result['has_ipd_sharing_plan'] = 0
        result['ipd_sharing_description'] = None
        result['ipd_sharing_info_types'] = None
        result['ipd_sharing_time_frame'] = None
        result['ipd_sharing_access_criteria'] = None
        result['ipd_sharing_url'] = None
    
    # ========================================================================
    # REFERENCES MODULE
    # ========================================================================
    references_module = safe_get(protocol, 'referencesModule', {})
    if isinstance(references_module, dict):
        references = safe_get(references_module, 'references', default=[])
        if isinstance(references, list):
            refs_list = []
            for ref in references:
                if isinstance(ref, dict):
                    refs_list.append({
                        'citation': safe_get(ref, 'citation'),
                        'type': safe_get(ref, 'type'),
                        'pmid': safe_get(ref, 'pmid'),
                        'url': safe_get(ref, 'url')
                    })
            result['references'] = json.dumps(refs_list) if refs_list else None
        else:
            result['references'] = None
        
        result['see_also_links'] = json.dumps(safe_get(references_module, 'seeAlsoLinks', default=[]))
        result['avail_ipds'] = json.dumps(safe_get(references_module, 'availIpds', default=[]))
    else:
        result['references'] = None
        result['see_also_links'] = None
        result['avail_ipds'] = None
    
    # ========================================================================
    # DERIVED SECTION - MeSH TERMS
    # ========================================================================
    derived_section = safe_get(record, 'derivedSection', {})
    
    # Condition browse (MeSH terms)
    mesh_terms = []
    if isinstance(derived_section, dict):
        condition_browse = safe_get(derived_section, 'conditionBrowseModule', {})
        if isinstance(condition_browse, dict):
            browse_leaves = safe_get(condition_browse, 'browseLeaves', default=[])
            if isinstance(browse_leaves, list):
                for leaf in browse_leaves:
                    if isinstance(leaf, dict):
                        mesh_term = (
                            safe_get(leaf, 'meshTerm') or
                            safe_get(leaf, 'term') or
                            safe_get(leaf, 'name') or
                            safe_get(leaf, 'title')
                        )
                        if mesh_term and isinstance(mesh_term, str):
                            mesh_terms.append(mesh_term)
                    elif isinstance(leaf, str):
                        mesh_terms.append(leaf)
    
    result['condition_mesh_terms'] = json.dumps(mesh_terms) if mesh_terms else None
    
    # Intervention browse (MeSH terms)
    intervention_mesh_terms = []
    if isinstance(derived_section, dict):
        intervention_browse = safe_get(derived_section, 'interventionBrowseModule', {})
        if isinstance(intervention_browse, dict):
            browse_leaves = safe_get(intervention_browse, 'browseLeaves', default=[])
            if isinstance(browse_leaves, list):
                for leaf in browse_leaves:
                    if isinstance(leaf, dict):
                        mesh_term = (
                            safe_get(leaf, 'meshTerm') or
                            safe_get(leaf, 'term') or
                            safe_get(leaf, 'name') or
                            safe_get(leaf, 'title')
                        )
                        if mesh_term and isinstance(mesh_term, str):
                            intervention_mesh_terms.append(mesh_term)
                    elif isinstance(leaf, str):
                        intervention_mesh_terms.append(leaf)
    
    result['intervention_mesh_terms'] = json.dumps(intervention_mesh_terms) if intervention_mesh_terms else None
    
    # ========================================================================
    # TOP-LEVEL FIELDS
    # ========================================================================
    result['has_results'] = safe_get(record, 'hasResults', False)
    
    # ========================================================================
    # ENROLLMENT TYPE HANDLING (prevent leakage)
    # ========================================================================
    # If enrollment_type is ACTUAL, set enrollment_planned to None
    if result.get('enrollment_planned_type') and str(result.get('enrollment_planned_type', '')).upper() == "ACTUAL":
        result['enrollment_planned'] = None
    
    # ========================================================================
    # CREATE LABEL (feasibility)
    # ========================================================================
    # IMPORTANT: Only label trials with FINAL status (completed or terminated).
    # Exclude ongoing trials (RECRUITING, ACTIVE_NOT_RECRUITING, etc.) because
    # their final outcome is unknown - they could still complete or terminate.
    status = result.get('overall_status', '').upper()
    if status == 'COMPLETED':
        result['label_feasible'] = 1
    elif status in ['TERMINATED', 'WITHDRAWN', 'SUSPENDED']:
        result['label_feasible'] = 0
    else:
        # All other statuses (ongoing, unknown, etc.) get None
        # These will be excluded from modeling
        result['label_feasible'] = None
    
    return result


def main():
    """Main parsing function."""
    print("=" * 80)
    print("Step 1: Parsing Raw ClinicalTrials.gov JSON Data from v2 API")
    print("=" * 80)
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    data_raw_v2_dir = project_root.parent / "data_raw" / "ctgov_api_v2"
    
    # Try to find JSONL files
    full_jsonl = data_raw_v2_dir / "oncology_phase234_interventional_industry_full_raw.jsonl"
    search_jsonl = data_raw_v2_dir / "oncology_phase234_interventional_industry_search_raw.jsonl"
    
    jsonl_file = None
    use_full_records = False
    
    if full_jsonl.exists():
        jsonl_file = full_jsonl
        use_full_records = True
        print(f"\nUsing full records from: {jsonl_file.name}")
    elif search_jsonl.exists():
        jsonl_file = search_jsonl
        use_full_records = False
        print(f"\nUsing search results from: {jsonl_file.name}")
    else:
        print(f"\nERROR: No JSONL files found in {data_raw_v2_dir}")
        print("Expected files:")
        print(f"  - {full_jsonl.name}")
        print(f"  - {search_jsonl.name}")
        return
    
    # Parse all trials
    print("\nParsing trials from JSONL file...")
    parsed_trials = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i % 1000 == 0:
                print(f"  Processed {i:,} trials...")
            
            try:
                data = json.loads(line.strip())
                
                if use_full_records:
                    # Full records have {"nct_id": "...", "record": {...}}
                    nct_id = data.get('nct_id')
                    record = data.get('record', {})
                    if not record:
                        record = data  # Fallback if structure is different
                else:
                    # Search results are just the study object
                    record = data
                    nct_id = None
                
                trial_data = parse_trial_record(record, nct_id)
                if trial_data:
                    parsed_trials.append(trial_data)
            except json.JSONDecodeError as e:
                print(f"  Error parsing line {i}: {e}")
                continue
            except Exception as e:
                print(f"  Error processing line {i}: {e}")
                continue
    
    print(f"\n[OK] Parsed {len(parsed_trials):,} trials matching filters")
    
    # Create DataFrame
    df = pd.DataFrame(parsed_trials)
    
    # Save to parquet
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / "01_parsed_trials.parquet"
    df.to_parquet(output_path, index=False)
    
    print(f"\n[OK] Saved parsed data to: {output_path}")
    print(f"  Shape: {df.shape[0]:,} trials × {df.shape[1]:,} columns")
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Label distribution:")
    if 'label_feasible' in df.columns:
        label_counts = df['label_feasible'].value_counts()
        print(f"    Completed (1): {label_counts.get(1, 0):,}")
        print(f"    Not Completed (0): {label_counts.get(0, 0):,}")
        ongoing_count = df['label_feasible'].isna().sum()
        print(f"    Ongoing/Unknown (NaN - will be excluded): {ongoing_count:,}")
        
        # Validation: Check that ongoing statuses have NaN labels
        if 'overall_status' in df.columns:
            ongoing_statuses = ['RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION', 
                               'NOT_YET_RECRUITING', 'UNKNOWN']
            ongoing_trials = df[df['overall_status'].isin(ongoing_statuses)]
            if len(ongoing_trials) > 0:
                ongoing_with_label = ongoing_trials[ongoing_trials['label_feasible'].notna()]
                if len(ongoing_with_label) > 0:
                    print(f"\n    ⚠️  WARNING: {len(ongoing_with_label)} ongoing trials have labels (should be NaN)")
                else:
                    print(f"\n    ✓ Validation: All {len(ongoing_trials):,} ongoing trials correctly have NaN labels")
    
    if 'phase' in df.columns:
        print(f"  Phases: {df['phase'].value_counts().to_dict()}")
    if 'study_type' in df.columns:
        print(f"  Study types: {df['study_type'].value_counts().to_dict()}")
    if 'lead_sponsor_type' in df.columns:
        print(f"  Sponsor types: {df['lead_sponsor_type'].value_counts().to_dict()}")
    
    print(f"\n[OK] Parsing complete!")


if __name__ == "__main__":
    main()
