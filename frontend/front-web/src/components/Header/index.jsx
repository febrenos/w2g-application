import React, {useState} from 'react'
// import styled from 'styled-components';

import {NavLink} from 'react-router-dom'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {faUserAlt, faChevronRight, faPaperPlane, faCompass, faBell, faGear, faBookmark, faChartSimple, faRightFromBracket, faHome, faCalendar, faCalendarMinus, faCalendarTimes, faListAlt, faTableList } from '@fortawesome/free-solid-svg-icons';//faChevronCircleRight
import './style.css'

export default function(props){
    return(
        <div>
            <ul className={props.isOpen ? 'menuOpen' : 'menuClose'}>
                <li onClick={props.onClick}><FontAwesomeIcon className="icon" icon={faChevronRight} id={props.isOpen ? 'iMenuOpen' : 'iMenuClose'}/></li>
                <NavLink to={"/"}>
                    <li><FontAwesomeIcon className="icon" icon={faHome}/><a>Home</a></li>
                </NavLink>
                <NavLink to={"/Profile"}>
                    <li><FontAwesomeIcon className="icon" icon={faUserAlt}/><a>Profile</a></li>
                </NavLink>
                <NavLink to={"/ToExplore"}>
                    <li><FontAwesomeIcon className="icon" icon={faCompass}/><a>To explore</a></li>
                </NavLink>
                <NavLink to={"/ChatBot"}>
                    <li><FontAwesomeIcon className="icon" icon={faPaperPlane}/><a>Chat-Bot</a></li>
                </NavLink>
                <NavLink to={"/Notifications"}>
                    <li><FontAwesomeIcon className="icon" icon={faBell}/><a>Notifications</a></li>
                </NavLink>
                <NavLink to={"/Bookmarks"}>
                    <li><FontAwesomeIcon className="icon" icon={faTableList}/><a>Lista eventos</a></li>
                </NavLink>
                <NavLink to={"/Dashboard"}>
                    <li><FontAwesomeIcon className="icon" icon={faChartSimple}/><a>Dashboard</a></li>
                </NavLink>
                <NavLink to={"/Settings"}>
                    <li><FontAwesomeIcon className="icon" icon={faGear}/><a>Settings</a></li>
                </NavLink>
                <NavLink to={"/Apresentation"}>
                    <li><FontAwesomeIcon className="icon" icon={faRightFromBracket}/><a>Logout</a></li>
                </NavLink>
            </ul>
        </div>   
    )
}
